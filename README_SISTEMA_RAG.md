# Sistema RAG Multimodal Completo com Gemini

Este projeto implementa um sistema completo de Retrieval-Augmented Generation (RAG) multimodal que processa documentos PDF, extrai imagens, gera embeddings e realiza análises contextuais usando modelos Gemini do Google Cloud Vertex AI.

## 🚀 Funcionalidades

- **Processamento de PDFs**: Extração automática de texto e imagens
- **Geração de Embeddings**: Para texto e imagens usando modelos especializados
- **Busca por Similaridade**: Sistema robusto de busca usando embeddings
- **Análise Contextual**: Análise inteligente com modelos Gemini
- **Sistema de Citações**: Referências e fontes para todas as informações
- **Processamento Direto**: Análise de imagens de pasta sem PDF
- **Interface CLI**: Linha de comando completa com opções flexíveis

## 📋 Pré-requisitos

### 1. Google Cloud Setup
```bash
# Instalar Google Cloud CLI
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Configurar autenticação
gcloud auth login
gcloud config set project SEU_PROJECT_ID

# Habilitar APIs necessárias
gcloud services enable aiplatform.googleapis.com
gcloud services enable compute.googleapis.com
```

### 2. Dependências Python
```bash
# Instalar dependências
pip install google-cloud-aiplatform
pip install vertexai
pip install pandas numpy pillow pymupdf rich
pip install scikit-learn
```

### 3. Estrutura de Diretórios
```
projeto/
├── multimodal_rag_complete.py    # Script principal
├── config.py                    # Configurações
├── exemplo_uso.py               # Script de exemplo
├── images/                      # Pasta de imagens
├── map/                         # Pasta de PDFs
└── README.md                    # Este arquivo
```

## 🛠️ Instalação

1. **Clone ou baixe os arquivos**:
   ```bash
   # Baixar os arquivos necessários
   wget multimodal_rag_complete.py
   wget config.py
   wget exemplo_uso.py
   ```

2. **Configure o Google Cloud**:
   ```bash
   # Definir variáveis de ambiente
   export GOOGLE_APPLICATION_CREDENTIALS="caminho/para/sua/service-account-key.json"
   export PROJECT_ID="seu-project-id"
   ```

3. **Crie as pastas necessárias**:
   ```bash
   mkdir -p images/ map/
   ```

## 🎯 Uso Básico

### Execução Simples
```bash
python multimodal_rag_complete.py
```

### Com Extração de PDF
```bash
python multimodal_rag_complete.py --extract-pdf
```

### Com Análise Direta
```bash
python multimodal_rag_complete.py --direct-analysis --target-image B2_room.jpeg
```

### Configuração Personalizada
```bash
python multimodal_rag_complete.py \
  --project-id meu-projeto \
  --location us-east1 \
  --embedding-size 256 \
  --image-dir minhas-imagens/ \
  --verbose
```

## 📖 Opções da Linha de Comando

### Configurações do Projeto
- `--project-id`: ID do projeto Google Cloud
- `--location`: Localização do Vertex AI (padrão: us-central1)

### Configurações de Processamento
- `--embedding-size`: Tamanho do embedding (128, 256, 512, 1408)
- `--image-dir`: Diretório das imagens (padrão: images/)
- `--pdf-dir`: Diretório dos PDFs (padrão: map/)

### Opções de Execução
- `--extract-pdf`: Extrair imagens de PDFs antes do processamento
- `--direct-analysis`: Executar análise direta com Gemini
- `--target-image`: Nome da imagem alvo para análise (padrão: M3.jpeg)

### Opções de Debug
- `--verbose, -v`: Modo verboso
- `--dry-run`: Executar sem fazer alterações

## 🔧 Exemplos de Uso

### 1. Processamento Básico
```bash
# Processar imagens existentes na pasta images/
python multimodal_rag_complete.py
```

### 2. Extração + Processamento
```bash
# Extrair imagens do PDF e processar
python multimodal_rag_complete.py --extract-pdf
```

### 3. Análise Completa
```bash
# Extração + Processamento + Análise Direta
python multimodal_rag_complete.py --extract-pdf --direct-analysis
```

### 4. Configuração Avançada
```bash
# Usar configurações específicas
python multimodal_rag_complete.py \
  --project-id meu-projeto \
  --embedding-size 1408 \
  --extract-pdf \
  --direct-analysis \
  --target-image minha_imagem.jpg \
  --verbose
```

### 5. Teste de Configuração
```bash
# Verificar configurações sem executar
python multimodal_rag_complete.py --dry-run --extract-pdf
```

## 📊 Fluxo de Execução

1. **Inicialização**: Carrega modelos Gemini e de embedding
2. **Extração** (opcional): Extrai imagens de PDFs
3. **Processamento**: Gera embeddings e descrições das imagens
4. **Validação**: Testa busca por similaridade
5. **Análise Contextual**: Análise baseada em imagens similares
6. **Análise Direta** (opcional): Análise direta com Gemini

## 🔍 Saída do Sistema

### Durante a Execução
```
🚀 INICIANDO SISTEMA RAG MULTIMODAL COMPLETO
============================================================
🚀 INICIALIZANDO MODELOS...
==================================================
✅ Project ID obtido automaticamente: meu-projeto
📋 Project ID: meu-projeto
📍 Location: us-central1
✅ Vertex AI inicializado
✅ Modelos multimodais carregados
✅ Modelos de embedding carregados
✅ Variáveis globais configuradas

🎉 TODOS OS MODELOS INICIALIZADOS COM SUCESSO!
```

### Resultados de Similaridade
```
🎉 SUCESSO! Encontradas 3 imagens similares:
================================================================================

🖼️  RESULTADO 1:
  📈 Similaridade: 0.8542
  📁 Arquivo: pasta_images_B2_room.jpeg
  📂 Caminho: images/B2_room.jpeg
  📝 Descrição: Esta imagem mostra um plano de piso detalhado...

📊 ANÁLISE DE SCORES:
  - Score máximo: 0.8542
  - Score mínimo: 0.6234
  - Score médio: 0.7388
```

### Análise Contextual
```
🤖 ANÁLISE CONTEXTUAL COM GEMINI:
======================================================================
📋 PERGUNTA 1: Baseado nas imagens similares encontradas, o que você pode me dizer sobre esta imagem?
======================================================================
🤖 RESPOSTA CONTEXTUALIZADA:
Baseado no contexto das imagens similares e na análise visual desta imagem...
```

## 🐛 Solução de Problemas

### Erro de Autenticação
```bash
# Verificar autenticação
gcloud auth list
gcloud config get-value project

# Reconfigurar se necessário
gcloud auth login
gcloud config set project SEU_PROJECT_ID
```

### Erro de Quota
```bash
# Usar configurações com sleep
python multimodal_rag_complete.py --embedding-size 128
```

### Erro de Dependências
```bash
# Instalar dependências faltantes
pip install google-cloud-aiplatform vertexai pandas numpy pillow pymupdf rich
```

### Erro de Permissões
```bash
# Verificar permissões do projeto
gcloud projects get-iam-policy SEU_PROJECT_ID
```

## 📁 Estrutura de Arquivos Gerados

- `image_metadata_from_folder.pkl`: DataFrame com metadados das imagens
- `images/`: Pasta com imagens extraídas e processadas
- Logs de execução no console

## 🔧 Personalização

### Modificar Configurações
Edite o arquivo `config.py`:
```python
# Configurações do Google Cloud
PROJECT_ID = "seu-projeto"
LOCATION = "us-central1"

# Configurações de Embedding
EMBEDDING_SIZE = 512
TOP_N_TEXT = 10
TOP_N_IMAGE = 5
```

### Adicionar Novos Formatos
```python
# Em config.py
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif']
```

## 📚 Recursos Adicionais

- **Script de Exemplo**: `python exemplo_uso.py`
- **Documentação Vertex AI**: https://cloud.google.com/vertex-ai/docs
- **Modelos Gemini**: https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/gemini

## 🤝 Contribuição

Para contribuir com o projeto:
1. Faça um fork do repositório
2. Crie uma branch para sua feature
3. Implemente suas mudanças
4. Teste thoroughly
5. Submeta um pull request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo LICENSE para detalhes.

## 🆘 Suporte

Para suporte e dúvidas:
- Abra uma issue no repositório
- Consulte a documentação do Google Cloud Vertex AI
- Verifique os logs de execução para diagnóstico

---

**Desenvolvido com ❤️ usando Google Cloud Vertex AI e Gemini**
