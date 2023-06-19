import os
from langchain.chat_models import AzureChatOpenAI 
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import (ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate)
import azure.cognitiveservices.speech as speechsdk

from dotenv import load_dotenv 
load_dotenv("/home/shohei/.env", override=True)

# Azure OpenAI のキーとエンドポイントの設定
OPENAI_API_BASE = os.environ["AZURE_OPENAI_API_BASE"]
OPENAI_API_VERSION = "2023-05-15"
OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
DEPLOYMENT_NAME ="gpt-35-turbo"
# Azure Speech Services のキーと音声の設定
SPPECH_ENDPOINT_ID = os.environ["AZURE_SPEECH_ENDPOINT_ID"] # for Custom Neural Voice (CNV)
SPEECH_KEY = os.environ["AZURE_SPEECH_API_KEY"]
SPEECH_REGION = "eastus"
VOICETYPE = "Lite ShoheiNeural" # VOICETYPE = "ja-JP-AoiNeural", "NanamiNeural"

# 音声サービスのインスタンス作成
speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
speech_config.speech_recognition_language="ja-JP"
audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True) 
speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

speech_config.endpoint_id = SPPECH_ENDPOINT_ID # for CNV
speech_config.speech_synthesis_voice_name=VOICETYPE
audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

def stt():
    # STT 実行
    speech_recognition_result = speech_recognizer.recognize_once_async().get()
    
    if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return speech_recognition_result.text
    elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized: {}".format(speech_recognition_result.no_match_details))
    elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_recognition_result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))
            print("Did you set the speech resource key and region values?")

def tts(answer_txt):
    # TTS 実行
    speech_synthesis_result = speech_synthesizer.speak_text_async(answer_txt).get()

    if speech_synthesis_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        pass
    elif speech_synthesis_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_synthesis_result.cancellation_details
        print("Speech synthesis canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            if cancellation_details.error_details:
                print("Error details: {}".format(cancellation_details.error_details))
                print("Did you set the speech resource key and region values?")

if __name__ == "__main__":
    # AzureChatOpenAIクラスのインスタンス作成
    chat = AzureChatOpenAI(openai_api_base = OPENAI_API_BASE, 
                           openai_api_version = OPENAI_API_VERSION, 
                           openai_api_key = OPENAI_API_KEY, 
                           deployment_name= DEPLOYMENT_NAME, 
                           temperature=0.7)
    # 会話の履歴を保持するためのメモリオブジェクトを作成
    memory = ConversationBufferMemory(return_messages=True)

    # # システムメッセージ用のテンプレートを定義
    # with open("./template.txt", "r", encoding='utf-8') as f: 
    #     template = f.read() # 別ファイルにあるプロンプトを読み込み
    # with open("./context.txt", "r", encoding='utf-8') as f: 
    #     context = f.read() # 別ファイルにあるコンテキスト (これまでの会話/行動の要約)を読み込み
    # system_template = template + context
    system_template = """あなたは私の友達です。あなたの名前は永田 (ナガタ)です。私とは親しい友達のように会話してください。敬語を使わないで、カジュアルに話してください"""

    # 上記テンプレートを用いてプロンプトテンプレートを作成
    prompt = ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(system_template), 
                                               MessagesPlaceholder(variable_name="history"), 
                                               HumanMessagePromptTemplate.from_template("{input}")])
    # 会話用のチェーンを作成: 初期化時に、使用するチャットモデル、メモリオブジェクト、プロンプトテンプレートを指定
    conversation = ConversationChain(llm=chat, memory=memory, prompt=prompt)

    # ユーザからの初回コマンドを音声入力
    print("マイクに向かって喋ってください。")
    command  = stt() #input("You: ")

    while command != "exit": 
        print(f"あなたの言葉: {command}")         
        # Buddy に返答を生成させる
        response = conversation.predict(input=command)
        print(f"AIの返答: {response}")
        # AIの返答を音声で出力
        tts(response)

        print("マイクに向かって喋ってください。")
        command = stt() #command = input("You: ")