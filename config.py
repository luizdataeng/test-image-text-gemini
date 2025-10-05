# Configuração do Sistema RAG Multimodal
# ======================================

# Configurações do Google Cloud
PROJECT_ID = "gen-lang-client-0303567819"
LOCATION = "us-central1"

# Configurações de Embedding
EMBEDDING_SIZE = 512  # Opções: 128, 256, 512, 1408
TOP_N_TEXT = 10
TOP_N_IMAGE = 5

# Configurações de Processamento
CHARACTER_LIMIT = 1000
OVERLAP = 100

# Diretórios
IMAGE_SAVE_DIR = "images/"
PDF_FOLDER_PATH = "map/"

# Configurações de Modelo
MODEL_TEMPERATURE = 0.2
MAX_OUTPUT_TOKENS = 2048

# Configurações de Segurança (BLOCK_NONE para desenvolvimento)
SAFETY_SETTINGS = {
    "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE", 
    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"
}

# Formatos de imagem suportados
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

# Configurações de Sleep (para evitar problemas de quota)
ADD_SLEEP_AFTER_PAGE = False
SLEEP_TIME_AFTER_PAGE = 2
ADD_SLEEP_AFTER_DOCUMENT = False
SLEEP_TIME_AFTER_DOCUMENT = 2
