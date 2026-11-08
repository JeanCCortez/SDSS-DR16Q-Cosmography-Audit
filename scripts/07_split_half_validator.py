import numpy as np
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# ENGINE: SPLIT-HALF CROSS-VALIDATION (EXACT LINEAR METHOD)
# ==============================================================================

WP_FIXED = 18.0803  # Prior do Pantheon+
N_SPLITS = 100

def extract_exact_phase(ra, z, res, wp):
    """Extrai matematicamente a fase global absoluta (sem mínimos locais)"""
    X = z * np.cos(ra - wp * z)
    Y = z * np.sin(ra - wp * z)
    A_matrix = np.vstack([X, Y]).T
    coeffs, _, _, _ = np.linalg.lstsq(A_matrix, res, rcond=None)
    
    phi0_rad = np.arctan2(coeffs[1], coeffs[0])
    if phi0_rad < 0:
        phi0_rad += 2 * np.pi
    return phi0_rad

def extract_amplitude_on_frozen_phase(ra, z, res, phi0, wp):
    """Projeta os resíduos contra a fase congelada e calcula a significância analítica"""
    V = z * np.cos(ra - (phi0 + wp * z))
    
    # Amplitude por mínimos quadrados unidimensional
    D0 = np.sum(V * res) / np.sum(V * V)
    
    # Cálculo do erro padrão analítico da amplitude para obter o Sigma
    residuals_model = res - D0 * V
    deg_freedom = len(res) - 1
    sigma_residuals = np.sqrt(np.sum(residuals_model**2) / deg_freedom)
    error_D0 = sigma_residuals / np.sqrt(np.sum(V * V))
    
    significance = D0 / error_D0
    return significance

if __name__ == "__main__":
    print("========================================================================")
    print(f" EXECUTANDO: VALIDACAO CRUZADA SPLIT-HALF REAL ({N_SPLITS} SPLITS)")
    print("========================================================================")
    
    # 1. Carregar dados limpos
    try:
        archive = np.load("purified_residuals.npz")
        ra_real = archive['ra']
        z_real = archive['z']
        res_real = archive['res']
    except FileNotFoundError:
        print("[ERRO] Você precisa rodar o script '05_orthogonal_systematics_cleaning.py' primeiro!")
        exit()
        
    n_objects = len(res_real)
    print(f"[1/2] Amostra limpa carregada: {n_objects} objetos.")
    print(f"[2/2] Executando bisseções aleatórias cegas...")
    
    significances_validation = []
    
    # Semente fixa para reprodutibilidade estrita do sorteio
    np.random.seed(42)
    
    indices = np.arange(n_objects)
    
    for i in range(N_SPLITS):
        # Embaralha os índices aleatoriamente para criar conjuntos mutuamente exclusivos
        shuffled_indices = np.random.permutation(indices)
        half_size = n_objects // 2
        
        idx_A = shuffled_indices[:half_size]
        idx_B = shuffled_indices[half_size:]
        
        # Passo 1: Calibra a Fase apenas na Metade A (Treinamento)
        phi0_calibrated = extract_exact_phase(ra_real[idx_A], z_real[idx_A], res_real[idx_A], WP_FIXED)
        
        # Passo 2: Testar a significância da Amplitude na Metade B (Dados Ocultos/Validação)
        sigma_validation = extract_amplitude_on_frozen_phase(ra_real[idx_B], z_real[idx_B], res_real[idx_B], phi0_calibrated, WP_FIXED)
        significances_validation.append(sigma_validation)
        
    significances_validation = np.array(significances_validation)
    
    # 2. Estatística Descritiva Real dos Resultados
    median_sigma = np.median(significances_validation)
    mean_sigma = np.mean(significances_validation)
    min_sigma = np.min(significances_validation)
    max_sigma = np.max(significances_validation)
    fraction_above_5sigma = np.sum(significances_validation >= 5.0) / N_SPLITS * 100
    
    print("\n========================================================================")
    print(" RESULTADO REAL DA VALIDAÇÃO SPLIT-HALF (DADOS NÃO VISTOS)")
    print("========================================================================")
    print(f" Menor significância encontrada   : {min_sigma:.2f}σ")
    print(f" Maior significância encontrada   : {max_sigma:.2f}σ")
    print(f" Média dos Splits                 : {mean_sigma:.2f}σ")
    print("------------------------------------------------------------------------")
    print(f" MEDIANA REAL DA SIGNIFICÂNCIA (DATO DO PAPER): {median_sigma:.2f}σ")
    print(f" Porcentagem de Splits acima de 5.0σ         : {fraction_above_5sigma:.1f}%")
    print("========================================================================\n")