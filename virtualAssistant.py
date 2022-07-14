try:
    import random
    import json
    import pickle
    import numpy as np
    import nltk
    import pyscreenshot
    import pyttsx3
    import datetime
    import speech_recognition as sr
    import wikipedia
    import webbrowser as wb
    import psutil
    import pywhatkit
    import os
    from nltk.stem import WordNetLemmatizer
    from tensorflow.keras.models import load_model

    engine = pyttsx3.init()
    engine.setProperty('rate', 135)


    def command():
        list = ['che ore sono',
                'che giorno è oggi',
                'cerca su wikipedia...',
                'cerca su google...',
                'manda un messaggio',
                'disconnetti',
                'spegnimento',
                'riavvia',
                'metti una canzone',
                'aggiungi un promemoria',
                'leggi promemoria',
                'fai uno screenshot',
                'stato del processore',
                'stop']
        print("AI: Ecco i comandi che potrai usare: ")
        speak("Ecco i comandi che potrai usare: ")
        for i in list:
            print(i)


    def speak(audio):
        engine.say(audio)
        engine.runAndWait()


    def time():
        current_time = datetime.datetime.now().strftime("%H:%M")
        speak(current_time)
        print("AI: " + current_time)


    def date():
        year = int(datetime.datetime.now().year)
        month = int(datetime.datetime.now().month)
        day = int(datetime.datetime.now().day)
        current_date = str(day)+"/"+str(month)+"/"+str(year)
        speak(current_date)
        print(current_date)


    def greetings():
        current_time = datetime.datetime.now().strftime("%H")
        if current_time >= '5' or current_time <= "14":
            return "Buongiorno"

        elif current_time >= "15" or current_time <= "18":
            return "Buon pomeriggio"

        elif current_time >= "19" or current_time <= "4":
            return "Buona sera"


    def wish_me():
        print("AI: {}, sono il tuo assistente personale".format(greetings()))
        speak("{}, sono il tuo assistente personale".format(greetings()))
        print("AI: Possiamo semplicemente parlare oppure")
        speak("Possiamo semplicemente parlare oppure")
        command()
        print("AI: Come posso aiutarti?")
        speak("Come posso aiutarti?")


    def screenshot():
        img = pyscreenshot.grab()
        img.save('screenshot.png')
        print("AI: Fatto")
        speak("Fatto")


    def cpu():
        usage = str(psutil.cpu_percent())
        print('AI: La CPU è al ' + usage + '%')
        speak('La CPU è al ' + usage + '%')
        battery = str(psutil.sensors_battery().percent)
        print('AI: Il livello della batteria è al: ' + battery + '%')
        speak('Il livello della batteria è: ' + battery + '%')


    def wikipedia_search(message):
        print("AI: Fammi controllare...")
        speak("Fammi controllare")
        message = message.replace("cerca su wikipedia ", "")
        wikipedia.set_lang('it')
        result = wikipedia.summary(message, sentences=2)
        print(result)
        speak(result)


    def google(message):
        print("AI: Fammi controllare...")
        speak("Fammi controllare")
        message = message.replace("cerca su google ", "")
        wb.open("www.google.com\\search?q=" + message)


    def send_message():
        print('AI: A chi vuoi inviare il messaggio?')
        speak('A chi vuoi inviare il messaggio')
        number = take_command()
        print('AI: Cosa vorresti scrivere?')
        speak('Cosa vorresti scrivere')
        txt = take_command()
        pywhatkit.sendwhatmsg_instantly(number, txt)


    def play_song():
        print("AI: Che canzone?")
        speak("Che canzone?")
        song = take_command()
        print("AI: Riproduco: " + song)
        speak("Riproduco: " + song)
        pywhatkit.playonyt(song, True, True)


    def take_notes():
        print("AI: Cosa ti devo ricordare?")
        speak("Cosa ti devo ricordare")
        data = take_command()
        print("AI: Hai detto di ricordarti di: " + data)
        speak("hai detto di ricordarti di: " + data)
        remember = open('data.txt', 'w')
        remember.write(data)
        remember.close()


    def read_notes():
        remember = open('data.txt', 'r')
        print('AI: Avevi detto di ricordarti: ' + remember.read())
        speak('Avevi detto di ricordarti: ' + remember.read())

    lemmatizer = WordNetLemmatizer()
    intents = json.loads(open('intents.json', encoding='utf8').read())

    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
    model = load_model('chatbot_model.h5')


    def clean_up_sentence(sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
        return sentence_words


    def bag_of_words(sentence):
        sentence_words = clean_up_sentence(sentence)
        bag = [0] * len(words)
        for w in sentence_words:
            for i, word in enumerate(words):
                if word == w:
                    bag[i] = 1
        return np.array(bag)


    def predict_class(sentence):
        bow = bag_of_words(sentence)
        res = model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.25
        result = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

        result.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in result:
            return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
        return return_list


    def get_response(intents_list, intents_json):
        global result
        tag = intents_list[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
        return result


    def take_command():
        global message
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Ti sto ascoltando...")
            r.pause_threshold = 1
            audio = r.listen(source)
        try:
            print("Riconoscendo l'input vocale....")
            message = r.recognize_google(audio, language='it-IT')
            print("TU: " + message)
        except Exception:
            res = random.choice(["Non ho capito", "Scusa, potresti ripetere?", "Prova ad usare parole più semplici."])
            speak(res)
            print("AI: " + res)
            take_command()
        return message

    if __name__ == "__main__":
        wish_me()
        while True:
            message = take_command().lower()
            if 'che ore sono' in message:
                time()

            elif 'che giorno è oggi' in message:
                date()

            elif 'cerca su wikipedia' in message:
                wikipedia_search(message)

            elif 'cerca su google' in message:
                google(message)

            elif 'manda un messaggio' in message:
                send_message()

            elif 'disconnetti' in message:
                os.system("shutdown -l")

            elif 'spegnimento' in message:
                os.system("shutdown -s")

            elif 'riavvia' in message:
                os.system("shutdown -r")

            elif 'metti una canzone' in message:
                play_song()

            elif 'aggiungi un promemoria' in message:
                take_notes()

            elif 'leggi promemoria' in message:
                read_notes()

            elif 'fai uno screenshot' in message:
                screenshot()

            elif 'stato del processore' in message:
                cpu()

            elif 'cosa sai fare' in message:
                command()

            elif 'stop' in message:
                ints = predict_class(message)
                res = get_response(ints, intents)
                print("AI: " + res)
                quit(speak(res))

            else:
                ints = predict_class(message)
                res = get_response(ints, intents)
                speak(res)
                print("AI: " + res)

except Exception as e:
    print(e)
