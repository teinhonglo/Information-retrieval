import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import operator

def convList2Dict(oList):
    dDict = {}
    for rank_pair in oList:
        key = rank_pair[0]
        val = rank_pair[1]
        dDict[key] = val
    return dDict

def prepRankingList(sDict, qry_IDs = None):
    AP_list = []
    if qry_IDs:
        for ID in qry_IDs:
            AP_list.append(sDict[ID])
    else:
        ranking_results = sorted(sDict.items(), key=operator.itemgetter(1), reverse = True)
        qry_IDs = []
        for q_ID, AP in ranking_results:
            qry_IDs.append(q_ID)
            AP_list.append(AP)
            
    return qry_IDs, AP_list
        

# NRM (TD)
#NRM_TD = [['20048.query', 0.02699815272036045], ['20091.query', 0.7371091871091872], ['20001.query', 0.4538878048669359], ['20039.query', 0.6884519243793575], ['20088.query', 0.14678831436128656], ['20020.query', 1.0], ['20071.query', 0.8136586871184446], ['20076.query', 0.0757619063494264], ['20096.query', 0.003615723521755908], ['20005.query', 0.8148830409356724], ['20015.query', 0.7821541142355798], ['20002.query', 0.13182618937055565], ['20070.query', 0.9276997901313522], ['20013.query', 0.9666666666666666], ['20089.query', 0.4810788772630878], ['20023.query', 1.0]]

# NRM (SD)
NRM_SD = [['20048.query', 0.09020202020202021], ['20091.query', 0.6871693121693121], ['20001.query', 0.4276916588858314], ['20039.query', 0.7573537444550175], ['20088.query', 0.03656499744346615], ['20020.query', 0.689922480620155], ['20071.query', 0.825330087736266], ['20076.query', 0.05700859041207635], ['20096.query', 0.03335376940876111], ['20005.query', 0.8561272061272062], ['20015.query', 0.9472639739206615], ['20002.query', 0.14158581211830767], ['20070.query', 0.8642221865041255], ['20013.query', 1.0], ['20089.query', 0.31043574865869916], ['20023.query', 1.0]]

# BERT (TD)
#BERT_TD = [['20048.query', 0.08446969696969697], ['20091.query', 0.9765432098765433], ['20001.query', 0.9857170981505171], ['20039.query', 0.14393012239544267], ['20088.query', 0.028500548058792794], ['20020.query', 1.0], ['20071.query', 0.6359288344337879], ['20076.query', 0.35361833803244747], ['20096.query', 0.002187564600853116], ['20005.query', 0.8480122655122655], ['20015.query', 0.9910571751017763], ['20002.query', 0.15577678838803016], ['20070.query', 0.9965526431034762], ['20013.query', 1.0], ['20089.query', 0.9704231665770127], ['20023.query', 0.0163706657001329]]

# BERT (SD)
BERT_SD = [['20048.query', 0.3163934426229508], ['20091.query', 1.0], ['20001.query', 0.9724701283989101], ['20039.query', 0.1521707210175818], ['20088.query', 0.01507727310523839], ['20020.query', 0.4777777777777777], ['20071.query', 0.6533775394129993], ['20076.query', 0.1851207654406004], ['20096.query', 0.0013971877206680095], ['20005.query', 0.8480122655122655], ['20015.query', 0.9504357399429153], ['20002.query', 0.10540156672233439], ['20070.query', 0.9498663307132842], ['20013.query', 0.9666666666666666], ['20089.query', 0.9647194910352805], ['20023.query', 0.3026315789473684]]

# BERT + NRM (SD)
BERT_NRM_SD = [['20048.query', 0.15782828282828285], ['20091.query', 1.0], ['20001.query', 0.9825865826116641], ['20039.query', 0.7046355274580649], ['20088.query', 0.022772458237391918], ['20020.query', 0.8666666666666667], ['20071.query', 0.9026280483102488], ['20076.query', 0.11125348682623377], ['20096.query', 0.004369039711238671], ['20005.query', 0.9809090909090908], ['20015.query', 0.9945398640333897], ['20002.query', 0.3489109468268095], ['20070.query', 0.961301802757704], ['20013.query', 1.0], ['20089.query', 0.9255208466746928], ['20023.query', 1.0]]

# covert list to dict
#NRM_TD_dict = {}
NRM_SD_dict = convList2Dict(NRM_SD)
#BERT_TD_dict = {}
BERT_SD_dict = convList2Dict(BERT_SD)
BERT_NRM_SD_dict = convList2Dict(BERT_NRM_SD)

# AP list
NRM_IDs, NRM_SD_APs = prepRankingList(NRM_SD_dict)
NRM_IDs, BERT_SD_APs = prepRankingList(BERT_SD_dict, NRM_IDs)
NRM_IDs, BERT_NRM_SD_APs = prepRankingList(BERT_NRM_SD_dict, NRM_IDs)

print(NRM_SD_APs)
print(BERT_SD_APs)
print(BERT_NRM_SD_APs)

plt.figure(8)
plt.plot(range(len(BERT_SD_APs)), BERT_SD_APs, color='#FF8800', label='BERT')
plt.plot(range(len(NRM_SD_APs)), NRM_SD_APs, color='#0066FF', label='NRM')
plt.plot(range(len(BERT_NRM_SD_APs)), BERT_NRM_SD_APs, color='green', label='BERT + NRM')
plt.title('Average Precision (AP)')
plt.legend(loc='upper right')
plt.savefig('mAP.png',dpi=300,format='png')
