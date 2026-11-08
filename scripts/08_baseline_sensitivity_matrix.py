import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# SCRIPT 08
# BASELINE SENSITIVITY MATRIX (KINEMATIC ROBUSTNESS)
# ==============================================================================

CATALOG_PATH = r"E:\Documentos\rrt\art3\new\DR16Q_Superset_v3.fits"
WP_FIXED = 18.0803
N_SHUFFLES = 1000

# Equações Cinemáticas (Baselines Agnósticos)
def t4(z, a, b, c, d): return a + b*np.log10(z) + c*z + d*(z**2)
def t5(z, a, b, c, d, e): return a + b*np.log10(z) + c*z + d*(z**2) + e*(z**3)
def t6(z, a, b, c, d, e, f): return a + b*np.log10(z) + c*z + d*(z**2) + e*(z**3) + f*(z**4)
def p21(z, a, b, c, d): return (a + b*z + c*(z**2)) / (1.0 + d*z)
def p22(z, a, b, c, d, e): return (a + b*z + c*(z**2)) / (1.0 + d*z + e*(z**2))

models = {
    "Taylor-4": (t4, [20, 1, 1, 1]),
    "Taylor-5": (t5, [20, 1, 1, 1, 1]),
    "Taylor-6": (t6, [20, 1, 1, 1, 1, 1]),
    "Padé (2,1)": (p21, [20, 1, 1, 0.1]),
    "Padé (2,2)": (p22, [20, 1, 1, 0.1, 0.1])
}

def extract_exact_amplitude(ra, z, res, wp):
    X = z * np.cos(ra - wp * z)
    Y = z * np.sin(ra - wp * z)
    A_matrix = np.vstack([X, Y]).T
    coeffs, _, _, _ = np.linalg.lstsq(A_matrix, res, rcond=None)
    return np.sqrt(coeffs[0]**2 + coeffs[1]**2)

if __name__ == "__main__":
    print("========================================================================")
    print(" EXECUTANDO: MATRIZ DE SENSIBILIDADE COSMOGRÁFICA")
    print("========================================================================")
    
    print("[1/3] Carregando SDSS DR16Q...")
    hdul = fits.open(CATALOG_PATH)
    data = hdul[1].data
    
    z_all = data['Z']
    ra_all_deg = data['RA']
    ra_all = ra_all_deg * np.pi / 180.0
    dec_deg = data['DEC']
    
    # Banda i (índice 3)
    psfmag = data['PSFMAG']
    mi_all = psfmag[:, 3] if len(psfmag.shape) > 1 else psfmag
    err_all = data['PSFMAGERR'][:, 3] if len(data['PSFMAGERR'].shape) > 1 else data['PSFMAGERR']
    
    print("[2/3] Aplicando cortes de qualidade...")
    mask_z = (z_all >= 1.5) & (z_all <= 2.0) & np.isfinite(mi_all) & (mi_all > 0)
    coords = SkyCoord(ra=ra_all_deg[mask_z]*u.deg, dec=dec_deg[mask_z]*u.deg, frame='icrs')
    b_gal = coords.galactic.b.deg
    mask_b = np.abs(b_gal) > 20.0
    
    z_c = z_all[mask_z][mask_b]
    ra_c = ra_all[mask_z][mask_b]
    ra_deg_c = ra_all_deg[mask_z][mask_b]
    dec_c = dec_deg[mask_z][mask_b]
    mi_c = mi_all[mask_z][mask_b]
    err_c = err_all[mask_z][mask_b]
    b_gal_c = np.abs(b_gal)[mask_b]
    
    print(f"      Analisando {len(z_c)} quasares limpos...\n")
    print("[3/3] Iterando sobre expansões cinemáticas (isso levará alguns minutos)...\n")
    
    # Definição dos bins de histograma para Densidade
    ra_bins = np.arange(0, 360 + 2, 2)
    dec_bins = np.arange(-90, 90 + 2, 2)
    
    for name, (func, p0) in models.items():
        try:
            # 1. Ajuste do Baseline Longitudinal
            popt, _ = curve_fit(func, z_c, mi_c, p0=p0, maxfev=15000)
            res_raw = mi_c - func(z_c, *popt)
            
            # 2. Sigma clipping iterativo
            mask_clip = np.abs(res_raw) < 3.0 * np.std(res_raw)
            res_clean = res_raw[mask_clip]
            
            z_f = z_c[mask_clip]
            ra_f = ra_c[mask_clip]
            ra_deg_f = ra_deg_c[mask_clip]
            dec_f = dec_c[mask_clip]
            err_f = err_c[mask_clip]
            b_gal_f = b_gal_c[mask_clip]
            
            # 3. Construção da Matriz de Sistemáticas Instrumentais
            H, ra_edges, dec_edges = np.histogram2d(ra_deg_f, dec_f, bins=[ra_bins, dec_bins])
            ra_idx = np.clip(np.digitize(ra_deg_f, ra_edges) - 1, 0, len(ra_bins)-2)
            dec_idx = np.clip(np.digitize(dec_f, dec_edges) - 1, 0, len(dec_bins)-2)
            rho_cell_f = H[ra_idx, dec_idx]
            
            # [Constante, Densidade, Declinação, Latitude Galáctica, Erro_PSF]
            V_matrix = np.vstack([np.ones_like(res_clean), rho_cell_f, dec_f, b_gal_f, err_f]).T
            sys_coeffs, _, _, _ = np.linalg.lstsq(V_matrix, res_clean, rcond=None)
            res_purificados = res_clean - (V_matrix @ sys_coeffs)
            
            # 4. Extração da Amplitude Real
            D0_real = extract_exact_amplitude(ra_f, z_f, res_purificados, WP_FIXED)
            
            # 5. Avaliação de Significância via Monte Carlo
            null_amps = []
            np.random.seed(42)  # Semente fixa para cada modelo garantir comparações justas
            for _ in range(N_SHUFFLES):
                ra_shuff = np.random.permutation(ra_f)
                null_amps.append(extract_exact_amplitude(ra_shuff, z_f, res_purificados, WP_FIXED))
            
            mu = np.mean(null_amps)
            sig = np.std(null_amps)
            z_score = (D0_real - mu) / sig
            
            print(f" > {name:15}: Amp(D0) = {D0_real:.5f} | Null Mean = {mu:.5f} ± {sig:.5f} | Z-Score = {z_score:.2f}σ")
            
        except Exception as e:
            print(f" [ERRO] Falha ao processar {name}: {e}")
    print("\n========================================================================")