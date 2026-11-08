import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# SCRIPT 19
# EXACT AMPLITUDE CONSISTENCY (FIXED GLOBAL GEOMETRY)
# ==============================================================================

CATALOG_PATH = r"E:\Documentos\rrt\art3\new\DR16Q_Superset_v3.fits"
WP_FIXED = 18.0803  # Prior do Pantheon+

def agnostic_baseline(z, coeffs):
    """Taylor-4 baseline evaluation."""
    return coeffs[0] + coeffs[1]*np.log10(z) + coeffs[2]*z + coeffs[3]*(z**2)

def extract_amplitude(ra, z, res, phi0_global, wp):
    """
    Extrai a Amplitude (D0) projetando os resíduos contra o vetor direcional fixo.
    D0 = (V dot res) / (V dot V)
    """
    V = z * np.cos(ra - (phi0_global + wp * z))
    D0 = np.sum(V * res) / np.sum(V * V)
    return D0

if __name__ == "__main__":
    print("========================================================================")
    print(" REDSHIFT BIN AMPLITUDE CONSISTENCY (FIXED GEOMETRY)")
    print("========================================================================")
    
    # Semente fixa para garantir total reprodutibilidade
    np.random.seed(42)
    
    # 1. Carregar dados
    print("[1/4] Carregando SDSS DR16Q e isolando Banda 'i'...")
    hdul = fits.open(CATALOG_PATH)
    data = hdul[1].data
    
    z_all = data['Z']
    ra_all = data['RA'] * np.pi / 180.0
    dec_all = data['DEC']
    
    # Correção crítica aplicada: usar a banda 'i' (índice 3) e não a 'r' (índice 2)
    psfmag = data['PSFMAG']
    mi_all = psfmag[:, 3] if len(psfmag.shape) > 1 else psfmag
    
    # 2. Cortes de qualidade
    mask_z = (z_all >= 1.5) & (z_all <= 2.0) & np.isfinite(mi_all) & (mi_all > 0)
    coords = SkyCoord(ra=data['RA'][mask_z]*u.deg, dec=dec_all[mask_z]*u.deg, frame='icrs')
    b_gal = coords.galactic.b.deg
    mask_b = np.abs(b_gal) > 20.0
    
    z_clean = z_all[mask_z][mask_b]
    ra_clean = ra_all[mask_z][mask_b]
    mi_clean = mi_all[mask_z][mask_b]
    
    # 3. Baseline Linear
    print("[2/4] Aplicando baseline agnóstico de Taylor...")
    X_base = np.vstack([np.ones_like(z_clean), np.log10(z_clean), z_clean, z_clean**2]).T
    base_coeffs, _, _, _ = np.linalg.lstsq(X_base, mi_clean, rcond=None)
    res = mi_clean - agnostic_baseline(z_clean, base_coeffs)
    
    std_res = np.std(res)
    mask_clip = np.abs(res) < 3.0 * std_res
    
    z_f = z_clean[mask_clip]
    ra_f = ra_clean[mask_clip]
    res_f = res[mask_clip]
    
    # 4. Extração da Fase Global
    print("[3/4] Calculando a Geometria Global Absoluta (φ₀)...")
    X = z_f * np.cos(ra_f - WP_FIXED * z_f)
    Y = z_f * np.sin(ra_f - WP_FIXED * z_f)
    A_matrix = np.vstack([X, Y]).T
    coeffs, _, _, _ = np.linalg.lstsq(A_matrix, res_f, rcond=None)
    
    phi0_global = np.arctan2(coeffs[1], coeffs[0])
    if phi0_global < 0:
        phi0_global += 2 * np.pi
        
    D0_global = np.sqrt(coeffs[0]**2 + coeffs[1]**2)
    
    # 5. Fatiamento e Extração de Amplitude
    print("[4/4] Fatiando o Universo e testando a Amplitude (D₀)...")
    b1 = (z_f >= 1.50) & (z_f < 1.75)
    b2 = (z_f >= 1.75) & (z_f <= 2.00)
    
    D0_b1 = extract_amplitude(ra_f[b1], z_f[b1], res_f[b1], phi0_global, WP_FIXED)
    D0_b2 = extract_amplitude(ra_f[b2], z_f[b2], res_f[b2], phi0_global, WP_FIXED)
    
    # Cálculo de variação percentual
    diff_percent = abs(D0_b1 - D0_b2) / D0_global * 100
    
    print("\n========================================================================")
    print(" RESULTADO DA CONSISTÊNCIA DE AMPLITUDE (D₀)")
    print("========================================================================")
    print(f" Geometria Global Fixada         : φ₀ = {np.degrees(phi0_global):.2f}°")
    print(f" Amplitude Global (1.50 - 2.00)  : D₀ = {D0_global:.5f}")
    print("------------------------------------------------------------------------")
    print(f" Amplitude Bin 1 (1.50 - 1.75)   : D₀ = {D0_b1:.5f}")
    print(f" Amplitude Bin 2 (1.75 - 2.00)   : D₀ = {D0_b2:.5f}")
    print("------------------------------------------------------------------------")
    print(f" Variação Relativa da Amplitude  : {diff_percent:.1f}%")
    print("========================================================================\n")