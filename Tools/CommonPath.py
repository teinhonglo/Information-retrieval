
class CommonPath(object):
    def __init__(self, is_training, is_short, is_spoken, is_TDT2 = True):
        if is_training:
            self.qry_path = "../Corpus/TDT2/Train/XinTrainQryTDT2/QUERY_WDID_NEW"
            self.rel_path = "../Corpus/TDT2/Train/QDRelevanceTDT2_forHMMOutSideTrain"
            self.log_filename = "train."
        else:
            self.log_filename = "test."
            if is_short:
                self.log_filename += "short."
                self.qry_path = "../Corpus/TDT2/QUERY_WDID_NEW_middle"
            else:
                self.log_filename += "long."
                self.qry_path = "../Corpus/TDT2/QUERY_WDID_NEW"
            self.rel_path = "../Corpus/TDT2/AssessmentTrainSet/AssessmentTrainSet.txt"

        if is_spoken:
            self.log_filename += "spk.log"
            self.doc_path = "../Corpus/TDT2/Spoken_Doc"
        else:
            self.log_filename += "text.log"
            self.doc_path = "../Corpus/TDT2/SPLIT_DOC_WDID_NEW"

        self.dict_path = "../Corpus/TDT2/LDC_Lexicon.txt"
        self.bg_path = "../Corpus/background"

    def getQryPath(self):
        return self.qry_path

    def getDocPath(self):
        return self.doc_path
    
    def getRelPath(self):
        return self.rel_path

    def getLogFilename(self):
        return self.log_filename

    def getDictPath(self):
        return self.dict_path
    
    def getBGPath(self):
        return self.bg_path
