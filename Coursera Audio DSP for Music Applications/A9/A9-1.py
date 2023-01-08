import soundDownload as SD
import soundAnalysis as SA

keyApi = "EtT7on8Da2JJDEUfF8CD0hyTCdY8BbDWjoq326Gs"
outputDir = "Data3"

#SD.downloadSoundsFreesound(queryText="guitar",
#API_Key=keyApi, outputDir="Target", topNResults=5,
#duration=(0,3), tag = "acoustic")
"""
SD.downloadSoundsFreesound(queryText="Kick",
API_Key=keyApi, outputDir=outputDir, topNResults=20,
duration=(0,3), tag = "one-shot")

SD.downloadSoundsFreesound(queryText="Snare",
API_Key=keyApi, outputDir=outputDir, topNResults=20,
duration=(0,3), tag = "one-shot")

SD.downloadSoundsFreesound(queryText="cymbal",
API_Key=keyApi, outputDir=outputDir, topNResults=20,
duration=(0,3), tag = "one-shot")
"""
"""
SD.downloadSoundsFreesound(queryText="violin",
API_Key=keyApi, outputDir=outputDir, topNResults=20,
duration=(0,3), tag = "pizzicato")

SD.downloadSoundsFreesound(queryText="trumpet",
API_Key=keyApi, outputDir=outputDir, topNResults=20,
duration=(0,3), tag = "single-note")

SD.downloadSoundsFreesound(queryText="cello",
API_Key=keyApi, outputDir=outputDir, topNResults=20,
duration=(0,3), tag = "multisample")
"""
amountOfCoeffUse = [3, 4]
#SA.descriptorPairScatterPlot(outputDir, descInput=amountOfCoeffUse)
#SA.showDescriptorMapping()

#SA.clusterSounds(outputDir, nCluster=3, descInput=amountOfCoeffUse)

inputSound = 'Target\\violin\\55997\\55997_692375-lq.json'
SA.classifySoundkNN(inputSound,
outputDir, 3, descInput=amountOfCoeffUse)

