from flask import Flask, request, render_template, redirect, url_for, Response
import psycopg2
import cv2
import base64
import os
import requests
from treino_com_db import *
from flask import Flask, request, render_template, jsonify 
from flask_caching import Cache


app = Flask(__name__, template_folder='templates', static_folder='static')

# Variável global para manter o contador de imagens
image_counter = 1

# Variáveis globais para armazenar os dados do formulário
nome = None
email = None
senha = None
confirmar_senha = None
restricao = None


app.config['CACHETYPE'] = 'SimpleCache'
app.config['CACHE_TYPE'] = 'MemcachedCache'
app.config['CACHE_MEMCACHED_SERVERS'] = ['127.0.0.1:5000']
cache = Cache(app)

cache_global = None

clf = cache.get('classificador.xml')
if clf is None:
    print("Modelo não encontrado no cache.", 500)

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

# Função para detectar e recortar o rosto na imagem
def face_cropped(img):
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.2, 5)

    if len(faces) > 1:
        faces = [faces[0]]

    if len(faces) == 0:
        return None

    for (x, y, w, h) in faces:
        cropped_face = img[y:y+h, x:x+w]
        return cropped_face

# Função para gerar frames da câmera
def gen_frames():
    camera = cv2.VideoCapture(1)  # Substitua pelo IP e porta do seu celular
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Redimensionar o frame para remover bordas pretas
            frame = cv2.resize(frame, (640, 480))
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def signup():
    return render_template('signup.html')  # Página inicial com o formulário

@app.route('/servidor_flask', methods=['POST'])
def servidor_flask():
    global nome, email, senha, confirmar_senha, restricao
    try:
        nome = request.form.get('nome')
        email = request.form.get('email')
        senha = request.form.get('senha')
        confirmar_senha = request.form.get('confirmar_senha')
        restricao = request.form.get('restricao')

        # Depuração: Verifica os dados recebidos
        print(f"Nome: {nome}")
        print(f"Email: {email}")
        print(f"Senha: {senha}")
        print(f"Confirmar Senha: {confirmar_senha}")
        print(f"Restrição: {restricao}")

        # Verificar se algum campo está vazio
        if not nome or not email or not senha or not confirmar_senha or not restricao:
            return "Todos os campos devem ser preenchidos.", 400

        # Verificar se as senhas coincidem
        if senha != confirmar_senha:
            return "As senhas não coincidem. Tente novamente.", 400

        # Conversão de restrição para inteiro
        try:
            restricao = int(restricao)  # Convertendo a restrição para inteiro
        except ValueError:
            return "Valor de restrição inválido.", 400

        # Conectar ao banco de dados PostgreSQL
        conn = connect_db()
        cursor = conn.cursor()

        # Inserir os dados no banco
        cursor.execute("""
            INSERT INTO cadastro (nome, email, senha, restricao)
            VALUES (%s, %s, %s, %s)
        """, (nome, email, senha, restricao))

        # Commit para salvar as alterações
        conn.commit()

        # Fechar a conexão
        cursor.close()
        conn.close()

        return redirect(url_for('cadastro_rosto'))  # Redireciona para a próxima página

    except Exception as e:
        # Captura e imprime o erro
        print(f"Erro: {str(e)}")
        return f"Erro no servidor: {str(e)}", 500

@app.route('/cadastro_rosto')
def cadastro_rosto():
    return render_template('cadastro_rosto.html')  # Aqui você pode ter a página de cadastro de rosto

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_image', methods=['POST'])
def capture_image():
    global image_counter, email
    camera = cv2.VideoCapture(1)  # Substitua pelo IP e porta do seu celular
    success, frame = camera.read()
    lg_id = request.form.get('lg_id')
    ids_imagens_bd = []
    if success:
        face = face_cropped(frame)
        if face is not None:
            face = cv2.resize(face, (400, 400))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            conn = connect_db()
            cursor = conn.cursor()

            # Consulta para obter o ID do usuário
            cursor.execute("SELECT lg_id FROM cadastro WHERE email = %s", (email,))
            id_usuario = cursor.fetchone()[0]

            # Gerar um nome de arquivo com a sequência numérica
            image_filename = f'{id_usuario}_{image_counter}.jpg'
            image_counter += 1

            # Converter a imagem processada em binário
            _, buffer = cv2.imencode('.jpg', face)
            image_binary = buffer.tobytes()

            for arquivo in arquivos:
                nome = arquivo["metadata"]["name"]
                id_imagem = arquivo["ipfs_pin_hash"]
                print(f"Nome: {nome}, ID: {id_imagem}")
                if f'{id_usuario}_' in nome:
                    ids_imagens_bd.append(id_imagem)

            classificador = treino_classificador(arquivos,id_usuario)


            @app.before_first_request
            def initialize_model():
                """Inicializa o modelo e armazena no cache."""
                global global_model
                clf = cv2.face.LBPHFaceRecognizer_create()  # Cria o modelo
                cache.set('classificador.xml', classificador)  # Armazena no cache
                global_model = classificador  # Define como variável global
                print("Modelo salvo no cache e como variável global.")

            @app.route('/predict', methods=['GET'])
            def predict():
                """Exemplo de rota para verificar o modelo."""
                global global_model

                # Recupera do cache se necessário
                if global_model is None:
                    print("Recuperando modelo do cache...")
                    global_model = cache.get('classificador')

                if global_model is None:
                    return "Modelo não encontrado no cache ou nas variáveis globais.", 500

                return "Modelo recuperado com sucesso, pronto para uso!"
            
            initialize_model (classificador)




            '''@app.beforefirstrequest
            def iniciar_modelo():
                global cache_global
                # Cria o modelo e armazena no cache
                cache.set('classificador.xml', classificador)
                cache_global = classificador
                print('Modelo salvo no cache')

                # Recupera o modelo do cache
               
                
            iniciar_modelo(classificador)

            @app.route('/predict', methods=['GET'])
            def predict():
                    # Recupera o modelo do cache
                    clf = cache.get('classificador.xml')
                    if clf is None:
                        return "Modelo não encontrado no cache.", 500'''

            #Reparte arquivo binário para subir no BD 
            pedacos_clf = reparte_bin_por_tamanho(clf, 10)
            print(f"Tipo de dados sendo salvo no BD: {type(pedacos_clf[0])}")
            for p in pedacos_clf:
                insere_dados_bd(conexao,'rosto', p, 'url')

            # Enviar a imagem para o Pinata
            pinata_url = "https://api.pinata.cloud/pinning/pinFileToIPFS"
            headers = {
                "pinata_api_key": "268a27b75d86ac82ab27",
                "pinata_secret_api_key": "6c7c0b15cbb237cf50b324ee69bc5e6db0dada585a8eff5343d60b5301b437f7"
            }
            files = {
                'file': (image_filename, image_binary, 'image/jpeg')
            }
            response = requests.post(pinata_url, headers=headers, files=files)

            # Depuração: Verifica a resposta da API do Pinata
            print("Resposta da API do Pinata:", response.json())

            # Verificação adicional da resposta da API
            if response.status_code == 200:
                print("Imagem enviada com sucesso para o Pinata.")
            else:
                print("Erro ao enviar a imagem para o Pinata:", response.text)

            cursor.close()
            conn.close()

    camera.release()
    return redirect(url_for('cadastro_rosto'))

if __name__ == "__main__":
    app.run(debug=True)
