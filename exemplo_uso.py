#!/usr/bin/env python3
"""
Script de Exemplo para Sistema RAG Multimodal
==============================================

Este script demonstra como usar o sistema RAG multimodal completo
com diferentes configurações e cenários de uso.

Autor: Sistema Multimodal RAG
Data: 2025
"""

import os
import sys
import subprocess
from pathlib import Path

def executar_comando(comando, descricao):
    """Executa um comando e mostra o resultado"""
    print(f"\n🚀 {descricao}")
    print("="*60)
    print(f"Comando: {comando}")
    print("-"*60)
    
    try:
        resultado = subprocess.run(comando, shell=True, capture_output=True, text=True)
        
        if resultado.stdout:
            print("SAÍDA:")
            print(resultado.stdout)
        
        if resultado.stderr:
            print("ERRO:")
            print(resultado.stderr)
        
        if resultado.returncode == 0:
            print("✅ Comando executado com sucesso!")
        else:
            print(f"❌ Comando falhou com código {resultado.returncode}")
            
    except Exception as e:
        print(f"❌ Erro ao executar comando: {e}")

def verificar_dependencias():
    """Verifica se as dependências estão instaladas"""
    print("🔍 VERIFICANDO DEPENDÊNCIAS...")
    print("="*50)
    
    dependencias = [
        "google-cloud-aiplatform",
        "vertexai", 
        "pandas",
        "numpy",
        "pillow",
        "pymupdf",
        "rich"
    ]
    
    for dep in dependencias:
        try:
            __import__(dep.replace("-", "_"))
            print(f"✅ {dep}")
        except ImportError:
            print(f"❌ {dep} - Execute: pip install {dep}")

def verificar_estrutura():
    """Verifica se a estrutura de diretórios está correta"""
    print("\n📁 VERIFICANDO ESTRUTURA DE DIRETÓRIOS...")
    print("="*50)
    
    diretorios = ["images/", "map/"]
    
    for dir_path in diretorios:
        if os.path.exists(dir_path):
            arquivos = len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
            print(f"✅ {dir_path} - {arquivos} arquivos")
        else:
            print(f"❌ {dir_path} - Diretório não encontrado")
            print(f"   Criando diretório...")
            os.makedirs(dir_path, exist_ok=True)
            print(f"✅ {dir_path} criado")

def exemplo_basico():
    """Exemplo básico de execução"""
    print("\n📋 EXEMPLO BÁSICO")
    print("="*50)
    
    comando = "python multimodal_rag_complete.py"
    executar_comando(comando, "Execução básica do sistema RAG")

def exemplo_com_extracao():
    """Exemplo com extração de PDF"""
    print("\n📄 EXEMPLO COM EXTRAÇÃO DE PDF")
    print("="*50)
    
    comando = "python multimodal_rag_complete.py --extract-pdf"
    executar_comando(comando, "Execução com extração de imagens do PDF")

def exemplo_com_analise_direta():
    """Exemplo com análise direta"""
    print("\n🔍 EXEMPLO COM ANÁLISE DIRETA")
    print("="*50)
    
    comando = "python multimodal_rag_complete.py --direct-analysis --target-image B2_room.jpeg"
    executar_comando(comando, "Execução com análise direta da imagem")

def exemplo_configuracao_personalizada():
    """Exemplo com configuração personalizada"""
    print("\n⚙️ EXEMPLO COM CONFIGURAÇÃO PERSONALIZADA")
    print("="*50)
    
    comando = "python multimodal_rag_complete.py --embedding-size 256 --image-dir images/ --verbose"
    executar_comando(comando, "Execução com configurações personalizadas")

def exemplo_dry_run():
    """Exemplo de dry-run"""
    print("\n🔍 EXEMPLO DE DRY-RUN")
    print("="*50)
    
    comando = "python multimodal_rag_complete.py --dry-run --extract-pdf --direct-analysis"
    executar_comando(comando, "Dry-run para verificar configurações")

def main():
    """Função principal do script de exemplo"""
    print("🚀 SCRIPT DE EXEMPLO - SISTEMA RAG MULTIMODAL")
    print("="*60)
    
    # Verificar se o script principal existe
    if not os.path.exists("multimodal_rag_complete.py"):
        print("❌ Arquivo 'multimodal_rag_complete.py' não encontrado!")
        print("   Certifique-se de que o script principal está no mesmo diretório.")
        return
    
    # Verificações iniciais
    verificar_dependencias()
    verificar_estrutura()
    
    # Menu de opções
    print("\n📋 OPÇÕES DISPONÍVEIS:")
    print("="*50)
    print("1. Exemplo básico")
    print("2. Exemplo com extração de PDF")
    print("3. Exemplo com análise direta")
    print("4. Exemplo com configuração personalizada")
    print("5. Exemplo de dry-run")
    print("6. Executar todos os exemplos")
    print("0. Sair")
    
    try:
        opcao = input("\nEscolha uma opção (0-6): ").strip()
        
        if opcao == "1":
            exemplo_basico()
        elif opcao == "2":
            exemplo_com_extracao()
        elif opcao == "3":
            exemplo_com_analise_direta()
        elif opcao == "4":
            exemplo_configuracao_personalizada()
        elif opcao == "5":
            exemplo_dry_run()
        elif opcao == "6":
            print("\n🔄 EXECUTANDO TODOS OS EXEMPLOS...")
            exemplo_dry_run()
            exemplo_basico()
            exemplo_com_extracao()
            exemplo_com_analise_direta()
            exemplo_configuracao_personalizada()
        elif opcao == "0":
            print("👋 Saindo...")
            return
        else:
            print("❌ Opção inválida!")
            return
            
    except KeyboardInterrupt:
        print("\n⚠️  Execução interrompida pelo usuário")
        return
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        return
    
    print("\n🎉 Exemplos executados!")
    print("\n💡 DICAS:")
    print("- Use --help para ver todas as opções disponíveis")
    print("- Use --dry-run para testar configurações sem executar")
    print("- Use --verbose para mais detalhes na execução")
    print("- Certifique-se de ter configurado o Google Cloud corretamente")

if __name__ == "__main__":
    main()
