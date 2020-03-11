from pyAudioAnalysis import audioTrainTest as aT

class DetectorAgr:
    
    def procesar(self, ruta, corpus):
        userPath="C:/Users/bassoldier/pyAudioAnalysis/pyAudioAnalysis/"
        if corpus == 'Corpus Chileno':
            model=userPath + "gbSenticChile"
            print("Corpus Chileno")

        if corpus == 'Corpus Híbrido':
            model=userPath + "gbSentic"
            print("Corpus Híbrido")

        if corpus == 'Corpus Británico (SAVEE)':
            model=userPath + "gbSenticSavee"
            print("Corpus SAVEE")

        resultado=aT.fileClassification(ruta, model,"gradientboosting")

        if resultado[0]==0.0:
            sentimiento="AGRESIVO"
        else:
            sentimiento="NO AGRESIVO"

        print(resultado)
        print(model)
        print("\n Este audio es catalogado como: " + sentimiento + "\n")
        print("Posee un "+ str(resultado[1][0]) + " de Agresividad y un " + str(resultado[1][1]) + " de NO AGRESIVIDAD")

        return resultado