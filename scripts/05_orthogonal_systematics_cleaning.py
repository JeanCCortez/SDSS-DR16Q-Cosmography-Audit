import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# ORTHOGONAL SYSTEMATICS CLEANING (EXACT LINEAR METHOD)
# ==============================================================================

CATALOG_PATH = r"E:\Documentos\rrt\art3\new\DR16Q_Superset_v3.fits"

def agnostic_baseline(z, coeffs):
    return coeffs[0] + coeffs[1]*np.log10(z) + coeffs[2]*z + coeffs[3]*(z**2)

if __name__ == "__main__":
    print("========================================================================")
    print(" EXECUTANDO: LIMPEZA ORTOGONAL DE SISTEMÁTICAS REAIS")
    print("========================================================================")
    
    # Semente fixa para garantir total reprodutibilidade
    np.random.seed(42)

    # 1. Carregar Dados Brutos
    print("[1/5] Carregando SDSS DR16Q...")
    hdul = fits.open(CATALOG_PATH)
    data = hdul[1].data
    
    z_all = data['Z']
    ra_all_deg = data['RA']
    ra_all = ra_all_deg * np.pi / 180.0
    dec_all = data['DEC']
    
    # Isolar banda i rigorosamente (índice 3)
    psfmag = data['PSFMAG']
    mi_all = psfmag[:, 3] if len(psfmag.shape) > 1 else psfmag
    err_all = data['PSFMAGERR'][:, 3] if len(data['PSFMAGERR'].shape) > 1 else data['PSFMAGERR']
    
    # 2. Aplicar Filtros de Qualidade Geométricos do Artigo (|b| > 20)
    print("[2/5] Aplicando cortes de qualidade...")
    mask_z = (z_all >= 1.5) & (z_all <= 2.0) & np.isfinite(mi_all) & (mi_all > 0)
    coords = SkyCoord(ra=ra_all_deg[mask_z]*u.deg, dec=dec_all[mask_z]*u.deg, frame='icrs')
    b_gal = coords.galactic.b.deg
    mask_b = np.abs(b_gal) > 20.0
    
    z_c = z_all[mask_z][mask_b]
    ra_c = ra_all[mask_z][mask_b]
    ra_deg_c = ra_all_deg[mask_z][mask_b]
    dec_c = dec_all[mask_z][mask_b]
    mi_c = mi_all[mask_z][mask_b]
    err_c = err_all[mask_z][mask_b]
    b_gal_c = np.abs(b_gal)[mask_b]  # Latitude galáctica absoluta
    
    # 3. Remover a tendência longitudinal via Taylor-4
    print("[3/5] Aplicando regressão longitudinal (Taylor-4)...")
    X_base = np.vstack([np.ones_like(z_c), np.log10(z_c), z_c, z_c**2]).T
    base_coeffs, _, _, _ = np.linalg.lstsq(X_base, mi_c, rcond=None)
    res_raw = mi_c - agnostic_baseline(z_c, base_coeffs)
    
    # Sigma clipping de 3.0 sigma para evitar outliers estocásticos
    std_init = np.std(res_raw)
    mask_clip = np.abs(res_raw) < 3.0 * std_init
    
    res_clean = res_raw[mask_clip]
    z_f = z_c[mask_clip]
    ra_f = ra_c[mask_clip]
    ra_deg_f = ra_deg_c[mask_clip]
    dec_f = dec_c[mask_clip]
    err_f = err_c[mask_clip]
    b_gal_f = b_gal_c[mask_clip]
    
    variancia_inicial = np.var(res_clean)
    
    # 4. Calcular Object Density (\rho_{cell}) per 2 degree cell
    print("[4/5] Calculando Densidade de Objetos (2° cells)...")
    ra_bins = np.arange(0, 360 + 2, 2)
    dec_bins = np.arange(-90, 90 + 2, 2)
    H, ra_edges, dec_edges = np.histogram2d(ra_deg_f, dec_f, bins=[ra_bins, dec_bins])
    
    ra_idx = np.digitize(ra_deg_f, ra_edges) - 1
    dec_idx = np.digitize(dec_f, dec_edges) - 1
    
    # Evitar out-of-bounds nas extremidades do array
    ra_idx = np.clip(ra_idx, 0, len(ra_bins)-2)
    dec_idx = np.clip(dec_idx, 0, len(dec_bins)-2)
    
    rho_cell_f = H[ra_idx, dec_idx]
    
    # 5. Construir a Matriz de Sistemáticas Instrumentais (V) conforme o Artigo
    # Colunas: [Constante, Densidade, Declinação, Latitude Galáctica, Erro_PSF]
    print("[5/5] Executando regressão ortogonal múltipla...")
    V_matrix = np.vstack([np.ones_like(res_clean), rho_cell_f, dec_f, b_gal_f, err_f]).T
    
    # Regressão Linear Múltipla para mapear a pegada do telescópio
    sys_coeffs, _, _, _ = np.linalg.lstsq(V_matrix, res_clean, rcond=None)
    res_predicted_sys = V_matrix @ sys_coeffs
    
    # Projeção Ortogonal: Subtrair o hiperplano de sistemáticas
    res_purificados = res_clean - res_predicted_sys
    variancia_final = np.var(res_purificados)
    
    # Calcular redução real de variância
    frac_removida = (variancia_inicial - variancia_final) / variancia_inicial * 100
    
    print("\n========================================================================")
    print(" RESULTADO REAL DA LIMPEZA DE SISTEMÁTICAS")
    print("========================================================================")
    print(f" Objetos analisados                 : {len(res_clean)}")
    print(f" Variância original dos resíduos    : {variancia_inicial:.6f}")
    print(f" Variância após limpeza ortogonal   : {variancia_final:.6f}")
    print("------------------------------------------------------------------------")
    print(f" PERCENTUAL REAL DE VARIÂNCIA REMOVIDO: {frac_removida:.2f}%")
    print("========================================================================\n")
    
    # Salva os resíduos purificados reais para os próximos passos
    np.savez("purified_residuals.npz", ra=ra_f, z=z_f, res=res_purificados)
    print("[OK] Resíduos purificados e limpos salvos em 'purified_residuals.npz'")