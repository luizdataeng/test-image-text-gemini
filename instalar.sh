#!/bin/bash
# Script de Instalação - Sistema RAG Multimodal
# ===============================================

echo "🚀 INSTALANDO SISTEMA RAG MULTIMODAL"
echo "===================================="

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Função para imprimir com cores
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Verificar se Python está instalado
echo "🔍 Verificando Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    print_status "Python $PYTHON_VERSION encontrado"
else
    print_error "Python3 não encontrado. Instale Python 3.8+ primeiro."
    exit 1
fi

# Verificar se pip está instalado
echo "🔍 Verificando pip..."
if command -v pip3 &> /dev/null; then
    print_status "pip3 encontrado"
else
    print_error "pip3 não encontrado. Instale pip primeiro."
    exit 1
fi

# Verificar se Google Cloud CLI está instalado
echo "🔍 Verificando Google Cloud CLI..."
if command -v gcloud &> /dev/null; then
    GCLOUD_VERSION=$(gcloud --version | head -n1 | cut -d' ' -f4)
    print_status "Google Cloud CLI $GCLOUD_VERSION encontrado"
else
    print_warning "Google Cloud CLI não encontrado"
    echo "Para instalar:"
    echo "  curl https://sdk.cloud.google.com | bash"
    echo "  exec -l \$SHELL"
fi

# Instalar dependências Python
echo "📦 Instalando dependências Python..."
print_info "Instalando pacotes necessários..."

pip3 install --upgrade pip

# Lista de dependências
DEPENDENCIES=(
    "google-cloud-aiplatform>=1.117.0"
    "vertexai"
    "pandas>=2.3.2"
    "numpy>=2.3.3"
    "pillow>=11.3.0"
    "pymupdf>=1.26.4"
    "rich>=14.1.0"
    "scikit-learn>=1.7.2"
    "requests"
)

for dep in "${DEPENDENCIES[@]}"; do
    echo "  Instalando $dep..."
    if pip3 install "$dep" --quiet; then
        print_status "$dep instalado"
    else
        print_error "Falha ao instalar $dep"
        exit 1
    fi
done

# Criar estrutura de diretórios
echo "📁 Criando estrutura de diretórios..."
mkdir -p images map
print_status "Diretórios criados: images/, map/"

# Tornar scripts executáveis
echo "🔧 Configurando permissões..."
chmod +x multimodal_rag_complete.py exemplo_uso.py
print_status "Scripts tornados executáveis"

# Verificar arquivos necessários
echo "🔍 Verificando arquivos do sistema..."
REQUIRED_FILES=(
    "multimodal_rag_complete.py"
    "config.py"
    "exemplo_uso.py"
    "README_SISTEMA_RAG.md"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        print_status "$file encontrado"
    else
        print_error "$file não encontrado"
        exit 1
    fi
done

# Verificar configuração do Google Cloud
echo "🔍 Verificando configuração do Google Cloud..."
if gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "@"; then
    ACTIVE_ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)")
    print_status "Conta ativa: $ACTIVE_ACCOUNT"
else
    print_warning "Nenhuma conta Google Cloud ativa encontrada"
    echo "Para configurar:"
    echo "  gcloud auth login"
    echo "  gcloud config set project SEU_PROJECT_ID"
fi

# Verificar projeto configurado
if gcloud config get-value project &> /dev/null; then
    PROJECT_ID=$(gcloud config get-value project)
    print_status "Projeto configurado: $PROJECT_ID"
else
    print_warning "Nenhum projeto Google Cloud configurado"
    echo "Para configurar:"
    echo "  gcloud config set project SEU_PROJECT_ID"
fi

# Teste básico do sistema
echo "🧪 Executando teste básico..."
if python3 multimodal_rag_complete.py --dry-run &> /dev/null; then
    print_status "Teste básico passou"
else
    print_warning "Teste básico falhou - verifique a configuração do Google Cloud"
fi

echo ""
echo "🎉 INSTALAÇÃO CONCLUÍDA!"
echo "========================"
echo ""
echo "📋 PRÓXIMOS PASSOS:"
echo "1. Configure o Google Cloud (se ainda não fez):"
echo "   gcloud auth login"
echo "   gcloud config set project SEU_PROJECT_ID"
echo ""
echo "2. Execute o sistema:"
echo "   python3 multimodal_rag_complete.py"
echo ""
echo "3. Para exemplos de uso:"
echo "   python3 exemplo_uso.py"
echo ""
echo "4. Para ajuda:"
echo "   python3 multimodal_rag_complete.py --help"
echo ""
echo "📚 Documentação completa: README_SISTEMA_RAG.md"
echo ""
print_status "Sistema RAG Multimodal instalado com sucesso!"
