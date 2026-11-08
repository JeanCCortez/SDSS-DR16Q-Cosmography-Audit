import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# SCRIPT 17
# INTERNAL CONSISTENCY CHECK: REDSHIFT SUB-INTERVALS
# ==============================================================================

CATALOG_PATH = r"E:\Documentos\rrt\art3\new\DR16Q_Superset_v3.fits"
WP_FIXED = 18.0803  # Prior do Pantheon+

def agnostic_baseline(z, a, b, c, d):
    """Taylor-4 expansion for longitudinal detrending."""
    return a + b*np.log10(z) + c*z + d*(z**2)

def precessing_dipole(data, D0, phi_0):
    """Dipole model with fixed precession."""
    ra, z = data
    return D0 * z * np.cos(ra - (phi_0 + WP_FIXED * z))

if __name__ == "__main__":
    print("========================================================================")
    print(" REDSHIFT BIN CONSISTENCY AUDIT")
    print("========================================================================")
    
    # Semente fixa para garantir total reprodutibilidade (como discutimos)
    np.random.seed(42)
    
    # 1. Carregar dados
    print("[1/4] Carregando SDSS DR16Q...")
    hdul = fits.open(CATALOG_PATH)
    data = hdul[1].data
    
    z_all = data['Z']
    ra_all = data['RA'] * np.pi / 180.0
    dec_all = data['DEC']
    
    # Extraindo a magnitude na banda 'i' rigorosamente (u=0, g=1, r=2, i=3, z=4)
    psfmag = data['PSFMAG']
    if len(psfmag.shape) > 1:
        mi_all = psfmag[:, 3]  # Agora usando a banda i correta
    else:
        mi_all = psfmag # Fallback de segurança
    
    # 2. Cortes de qualidade iniciais
    mask_z = (z_all >= 1.5) & (z_all <= 2.0) & np.isfinite(mi_all) & (mi_all > 0)
    
    # Corte Galáctico (|b| > 20)
    coords = SkyCoord(ra=data['RA'][mask_z]*u.deg, dec=dec_all[mask_z]*u.deg, frame='icrs')
    b_gal = coords.galactic.b.deg
    mask_b = np.abs(b_gal) > 20.0
    
    z_clean = z_all[mask_z][mask_b]
    ra_clean = ra_all[mask_z][mask_b]
    mi_clean = mi_all[mask_z][mask_b]
    
    print(f"      Objetos retidos após cortes geométricos: {len(z_clean)}")

    # 3. Baseline Agnóstico e Sigma Clipping
    print("[2/4] Aplicando baseline cosmográfico e 3.0σ clipping...")
    popt_base, _ = curve_fit(agnostic_baseline, z_clean, mi_clean)
    res = mi_clean - agnostic_baseline(z_clean, *popt_base)
    
    std_res = np.std(res)
    mask_clip = np.abs(res) < 3.0 * std_res
    
    z_final = z_clean[mask_clip]
    ra_final = ra_clean[mask_clip]
    res_final = res[mask_clip]
    
    # 4. Divisão dos Bins de Redshift
    print("[3/4] Dividindo em fatias de Redshift...")
    bin1_mask = (z_final >= 1.50) & (z_final < 1.75)
    bin2_mask = (z_final >= 1.75) & (z_final <= 2.00)
    
    print(f"      Bin 1 (1.50 - 1.75): {np.sum(bin1_mask)} objetos")
    print(f"      Bin 2 (1.75 - 2.00): {np.sum(bin2_mask)} objetos")
    
    # 5. Otimização Independente
    print("[4/4] Extraindo a fase transversal (φ₀) independente por bin...")
    
    # Fit Bin 1
    popt1, _ = curve_fit(precessing_dipole, (ra_final[bin1_mask], z_final[bin1_mask]), res_final[bin1_mask], p0=[0.01, 3.0])
    phi1_rad = popt1[1] % (2 * np.pi)
    phi1_deg = np.degrees(phi1_rad)
    
    # Fit Bin 2
    popt2, _ = curve_fit(precessing_dipole, (ra_final[bin2_mask], z_final[bin2_mask]), res_final[bin2_mask], p0=[0.01, 3.0])
    phi2_rad = popt2[1] % (2 * np.pi)
    phi2_deg = np.degrees(phi2_rad)
    
    # Calcula a menor diferença angular entre as duas fases
    diff_deg = abs(phi1_deg - phi2_deg)
    if diff_deg > 180:
        diff_deg = 360 - diff_deg
        
    print("\n========================================================================")
    print(" RESULTADO DA CONSISTÊNCIA ESTRUTURAL")
    print("========================================================================")
    print(f" Fase extraída no Bin 1 (1.50 <= z < 1.75) : {phi1_deg:.2f}°")
    print(f" Fase extraída no Bin 2 (1.75 <= z <= 2.00): {phi2_deg:.2f}°")
    print("------------------------------------------------------------------------")
    print(f" Diferença estrutural (Δφ₀)                : {diff_deg:.2f}°")
    print("========================================================================\n")