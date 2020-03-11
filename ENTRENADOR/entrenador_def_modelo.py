from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
from pyAudioAnalysis import MidTermFeatures
from pyAudioAnalysis import audioAnalysis
import matplotlib.pyplot as plt
from pyAudioAnalysis import audioTrainTest as aT

#aT.featureAndTrain(["SELECCION/AGRESIVOS/","SELECCION/NO_AGRESIVOS/"], 1.0, 0.10, aT.shortTermWindow, aT.shortTermStep, "svm", "svmSentic", False)
#aT.featureAndTrain(["SELECCION/AGRESIVOS/","SELECCION/NO_AGRESIVOS/"], 1.0, 0.10, 0.020, 0.020, "svm", "svmSentimiento", False)

aT.featureAndTrain(["SELECCION/AGRESIVOS/","SELECCION/NO_AGRESIVOS/"], 1.0, 0.10, 0.020, 0.020, "gradientboosting", "gbSentic", False)

#aT.featureAndTrain(["SELECCION/CHILE/AGRESIVOS/","SELECCION/CHILE/NO_AGRESIVOS/"], 1.0, 0.10, 0.020, 0.020, "gradientboosting", "gbSenticChile", False)

#aT.featureAndTrain(["SELECCION/SAVEE/AGRESIVOS/","SELECCION/SAVEE/NO_AGRESIVOS/"], 1.0, 0.10, 0.020, 0.020, "gradientboosting", "gbSenticSavee", False)



#print(aT.fileClassification("68.wav", "svmSentic","svm"))



#RANDOM FOREST
#aT.featureAndTrain(["SELECCION/AGRESIVOS/","SELECCION/NO_AGRESIVOS/"], 1.0, 0.10, 0.020, 0.020, "randomforest", "rfSentic", False)

#aT.featureAndTrain(["SELECCION/CHILE/AGRESIVOS/","SELECCION/CHILE/NO_AGRESIVOS/"], 1.0, 0.10, 0.020, 0.020, "randomforest", "gbSenticChile", False)

#aT.featureAndTrain(["SELECCION/SAVEE/AGRESIVOS/","SELECCION/SAVEE/NO_AGRESIVOS/"], 1.0, 0.10, 0.020, 0.020, "randomforest", "gbSenticSavee", False)




"""resultado=aT.fileClassification("voz.wav", "gbSentic","gradientboosting")

if resultado[0]==0.0:
	sentimiento="Agresivo"
else:
	sentimiento="No Agresivo"

print(sentimiento)
print(resultado)"""

