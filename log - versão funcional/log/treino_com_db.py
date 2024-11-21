import os
import cv2
from PIL import Image
import numpy as np
import psycopg2
import requests
from io import BytesIO

# Função para conectar ao banco de dados PostgreSQL
def connect_db():
    return psycopg2.connect(
        user="postgres",
        password="fouygAKVUZVhIxoeZxUJJrLYbXQjPdJd",
        host="junction.proxy.rlwy.net",  # Endereço do servidor
        port="39581",                    # Porta padrão do PostgreSQL
        database="railway",
        sslmode='disable',   
        connect_timeout=300,  # Aumente o tempo de conexão
        options="-c statement_timeout=0",  # Desabilitar o tempo de execução
        keepalives_idle=60,
    )


def consulta_bd(bd, query):
    try:
        with bd.cursor() as cursor:
            sql = query
            cursor.execute(sql)
            resultado = cursor.fetchall()
            print (f"Resultado da consulta: {resultado}")
    except Exception as e:
        print(f"Erro ao executar a query: {e}")

import requests

# Função para listar arquivos no Pinata com depuração
def listar_arquivos_pinata():
    url = "https://api.pinata.cloud/data/pinList"
    headers = {
        "pinata_api_key": "268a27b75d86ac82ab27",
        "pinata_secret_api_key": "6c7c0b15cbb237cf50b324ee69bc5e6db0dada585a8eff5343d60b5301b437f7"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        arquivos = response.json().get("rows", [])
        return arquivos
    else:
        print("Erro ao listar arquivos no Pinata:", response.text)
        return []

# Listar arquivos no Pinata
arquivos = listar_arquivos_pinata()




# Função para baixar e processar arquivos do Pinata
def processar_arquivo_pinata(hash):
    url = f"https://gateway.pinata.cloud/ipfs/{hash}"
    response = requests.get(url)
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content)).convert('L')
        image_np = np.array(image, 'uint8')
        image_gb = cv2.GaussianBlur(image_np, (5, 5), 0)
        return image_gb
    else:
        print(f"Erro ao baixar o arquivo {hash}:", response.text)
        return None


def reparte_bin_por_tamanho(bin, tamanho_max_mb):
    print("Iniciando repartição...")
    tamanho_max_bytes = tamanho_max_mb * 1024 * 1024
    tam_bin = len(bin)
    n_partes = tam_bin // tamanho_max_bytes
    if tam_bin % tamanho_max_bytes != 0:  # Se houver resto, adicione mais uma parte
        n_partes += 1

    print(f"O arquivo será dividido em {n_partes} partes, cada uma com até {tamanho_max_mb} MB.\n")

    tam_partes = tam_bin // n_partes
    partes = []
    cont = 0

    while True:
        if cont < n_partes:
            # Extrai a parte correspondente
            parte = bin[tam_partes * cont:tam_partes * (cont + 1)]
            if n_partes * tam_partes == tam_bin:
                resto = b''
            else:
                ini = tam_bin // n_partes * n_partes
                resto = bin[ini:]

            partes.append(parte)
            print(f"Tamanho parte {cont + 1}: {len(parte) / (1024 * 1024):.2f} MB")

            if n_partes * tam_partes < tam_bin and cont == (n_partes - 1):
                break

            if n_partes * tam_partes < tam_bin and cont == n_partes:
                break
            cont += 1
        else:
            break
        partes[-1] = partes[-1] + resto
    return partes



def treino_classificador(arquivos, id):

    rostos = []
    ids = []

    #verificador de mudança recebe primeiro id da lista
    for image in arquivos:
        if image.split()[-1] == "jpg":
            img = Image.open(image)
        else:
            img = Image.open(image).convert('L')
        
        #conversão para array
        imageNp = np.array(img, 'uint8')
        #Redução de ruido
        imageGB = cv2.GaussianBlur(imageNp, (5, 5), 0)

        #if id != id_ant:
            
        #Exibe imagens com filtro
        #cv2.imwrite(f"{id} Original.png", imageNp)
        #cv2.imwrite(f"{id} Red Ruido.png", imageGB)
        

        rostos.append(imageGB)
        ids.append(id)
    ids = np.array(ids)

    #nome_classficador = 'classificador.yml'
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(rostos,ids)
    #clf.write(nome_classficador)

    return clf

def insere_dados_bd(bd, tabela, campos, valores):
    try:
        '''
        #Verifica se o bd ainda está conectado (Rail)
        if not bd.isexecuting():
            print("Reconectando ao Bando de dados...")
            bd.reconnect(attempts=3, delay=5)
            if bd.isexecuting():
                print("Reconexão feita com sucesso!")
        '''
        with bd.cursor() as cursor:

            #dados = {'url': 'teste'}
            campos_str = "(" + ", ".join(campos) + ")"
            # Monta os placeholders para os valores
            placeholders = "(" + ", ".join(["?" for _ in campos]) + ")"
            # Cria a query
            sql = f"INSERT INTO {tabela} {campos_str} VALUES {placeholders}"

            cursor.execute(sql, valores)
            print('Dados inseridos com sucesso!')
            bd.commit()
    except Exception as e:
        print(f"Erro cadastrar dados: {e}")

# Conectar ao banco de dados
conexao = connect_db()

# Listar arquivos no Pinata
arquivos = listar_arquivos_pinata()

rostos = []
ids = []



#BANCO DE DADOS
#Exibe BD antes e depois do cadastro
#Exibe colunas e variaveis aceitas
#acesa arquivo local e salva no BD
'''
t_ini = time.time()

with open('classificador.yml', 'rb') as f:
    clf_txt = f.read()

#Reparte arquivo binário para subir no BD 
pedacos_clf = reparte_bin(clf_txt, 14)
print(f"Tipo de dados sendo salvo no BD: {type(pedacos_clf[0])}")
for p in pedacos_clf:
    insere_dados_bd(conexao,nome_tabela, p)

t_fin = time.time()
diferenca_tempo(t_ini, t_fin, "Tempo de inclusão no BD")'''




