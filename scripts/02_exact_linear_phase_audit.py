import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# SCRIPT 18
# EXACT LINEAR ALGEBRA BIN CONSISTENCY
# ==============================================================================

CATALOG_PATH = r"E:\Documentos\rrt\art3\new\DR16Q_Superset_v3.fits"
WP_FIXED = 18.0803  # Prior do Pantheon+

def agnostic_baseline(z, coeffs):
    """Taylor-4 baseline evaluation."""
    return coeffs[0] + coeffs[1]*np.log10(z) + coeffs[2]*z + coeffs[3]*(z**2)

def extract_exact_dipole(ra, z, res, wp):
    """
    Extrator Algébrico Linear Perfeito para o Dipolo.
    Sem mínimos locais. Encontra a fase global absoluta matematicamente.
    """
    X = z * np.cos(ra - wp * z)
    Y = z * np.sin(ra - wp * z)
    
    # Monta a matriz para regressão linear: res = A*X + B*Y
    A_matrix = np.vstack([X, Y]).T
    coeffs, _, _, _ = np.linalg.lstsq(A_matrix, res, rcond=None)
    
    A, B = coeffs[0], coeffs[1]
    
    # Reconversão para Amplitude e Fase
    D0 = np.sqrt(A**2 + B**2)
    phi0_rad = np.arctan2(B, A)
    
    if phi0_rad < 0:
        phi0_rad += 2 * np.pi
        
    return D0, np.degrees(phi0_rad)

if __name__ == "__main__":
    print("========================================================================")
    print(" REDSHIFT BIN CONSISTENCY (EXACT LINEAR METHOD)")
    print("========================================================================")
    
    # Semente fixa para garantir total reprodutibilidade
    np.random.seed(42)
    
    # 1. Carregar dados
    print("[1/4] Carregando SDSS DR16Q...")
    hdul = fits.open(CATALOG_PATH)
    data = hdul[1].data
    
    z_all = data['Z']
    ra_all = data['RA'] * np.pi / 180.0
    dec_all = data['DEC']
    
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
    
    # 4. Fatiamento
    print("[3/4] Fatiando o Universo...")
    b1 = (z_f >= 1.50) & (z_f < 1.75)
    b2 = (z_f >= 1.75) & (z_f <= 2.00)
    
    # 5. Extração Matemática
    print("[4/4] Resolvendo matrizes lineares para φ₀...")
    
    D_full, phi_full = extract_exact_dipole(ra_f, z_f, res_f, WP_FIXED)
    D_b1, phi_b1 = extract_exact_dipole(ra_f[b1], z_f[b1], res_f[b1], WP_FIXED)
    D_b2, phi_b2 = extract_exact_dipole(ra_f[b2], z_f[b2], res_f[b2], WP_FIXED)
    
    diff_deg = abs(phi_b1 - phi_b2)
    if diff_deg > 180:
        diff_deg = 360 - diff_deg
        
    print("\n========================================================================")
    print(" RESULTADO DA CONSISTÊNCIA ESTRUTURAL EXATA (MÍNIMOS QUADRADOS)")
    print("========================================================================")
    print(f" Catálogo Completo (1.50 - 2.00) : φ₀ = {phi_full:.2f}°  (Amp = {D_full:.5f})")
    print("------------------------------------------------------------------------")
    print(f" Bin 1 (1.50 <= z < 1.75)        : φ₀ = {phi_b1:.2f}°  (Amp = {D_b1:.5f})")
    print(f" Bin 2 (1.75 <= z <= 2.00)       : φ₀ = {phi_b2:.2f}°  (Amp = {D_b2:.5f})")
    print("------------------------------------------------------------------------")
    print(f" Diferença Estrutural Real (Δφ₀) : {diff_deg:.2f}°")
    print("========================================================================\n")