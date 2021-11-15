# This is the file you will need to edit in order to complete assignment 1
# You may create additional functions, but all code must be contained within this file


# Some starting imports are provided, these will be accessible by all functions.
# You may need to import additional items
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas.core.series import Series

# You should use these two variable to refer the location of the JSON data file and the folder containing the news articles.
# Under no circumstances should you hardcode a path to the folder on your computer (e.g. C:\Chris\Assignment\data\data.json) as this path will not exist on any machine but yours.
datafilepath = 'data/data.json'
articlespath = 'data/football'

def task1():
    # load Json file
    import json
    with open(datafilepath, encoding="utf8") as f:
        Data = json.load(f)
        teams_codes = Data["teams_codes"]
        sort = sorted(teams_codes)
    return sort
    
def task2():
    import json
    club_code = []
    goal_scored = []
    goal_conceded = []
    with open(datafilepath, encoding="utf8") as f:
        Data = json.load(f)
        for clubs in Data["clubs"]:
            club_code.append(clubs['club_code'])
            goal_scored.append(clubs['goals_scored'])
            goal_conceded.append(clubs['goals_conceded'])

        clubcode = pd.Series(club_code)
        goalscored = pd.Series(goal_scored)
        goalconceded = pd.Series(goal_conceded)
        csv = pd.DataFrame({"team_code":clubcode,"goals_scored_by_team":goalscored,"goals_scored_against_team":goalconceded})
        csv = csv.set_index('team_code')
        csv = csv.sort_values(by="team_code",ascending=True)
        csv.to_csv('task2.csv')
    return
      
def task3():
    import os
    import re
    scorelist = []
    filenamelist = []
    for filename in os.listdir(articlespath):
        if filename.endswith(".txt"):
            largestMatchScore=0
            filenamelist.append(filename)
            with open(articlespath+ '/' +filename, encoding="utf8") as f:
                text = f.read()
                pattern = '(?<![0-9])(\d{1,2}-\d{1,2})(?![0-9])'
                if re.search(pattern, text) :
                    s=re.findall(pattern, text)
                    #s is a list of scores in that article
                    #scores is a score eg: 12-56
                    for scores in s:
                        if re.search('(\d{1,2})', scores):
                            num = re.findall('(\d{1,2})', scores)
                            max = int(num[0])+int(num[1])
                            if(int(max)>largestMatchScore):
                                largestMatchScore = max
                    #print(largestMatchScore)
                    scorelist.append(largestMatchScore)

                else :
                    scorelist.append(0)
    scorelist = pd.Series(scorelist)
    filenamelist = pd.Series(filenamelist)
    csv = pd.DataFrame({"filename":filenamelist,"total_goals":scorelist})
    csv = csv.sort_values(by="filename",ascending=True)
    csv = csv.set_index('filename')
    csv.to_csv('task3.csv')
    
    return

def task4():
    csv = pd.read_csv('task3.csv')
    csv = csv.set_index('filename')
    plt.figure(figsize=(10,10))
    plt.boxplot(csv)
    plt.xlabel("Matches")
    plt.ylabel("Total goals")
    plt.suptitle('Total goals box plot', fontsize=20, fontweight='bold')
    plt.xlim(0,2)
    plt.ylim(0, 105)
    plt.savefig('task4.png')
    plt.close()
    return
    
def task5():
    import json
    import os
    from numpy import arange
    name = []
    clubnumberlist = []
    with open(datafilepath, encoding="utf8") as f:
        Data = json.load(f)
        for clubs in Data["clubs"]:
            name.append(clubs['name'])
    for club in name:
        clubnumber = 0
        for filename in os.listdir(articlespath):
            if filename.endswith(".txt"):
                with open(articlespath+ '/' +filename, encoding="utf8") as f:
                    text = f.read()
                    if(text.count(club)>0):
                        clubnumber=clubnumber+1 
                    #print(filename+str(clubnumber))
        clubnumberlist.append(clubnumber)
    clubnumber = pd.Series(clubnumberlist)
    name = pd.Series(name)
    csv = pd.DataFrame({"club_name":name,"number_of_mentions":clubnumber})
    csv = csv.sort_values(by="club_name",ascending=True)
    csv = csv.set_index('club_name')
    csv.to_csv('task5.csv')

    #produce bar chart
    plt.bar(arange(len(clubnumberlist)),clubnumberlist)
    plt.xticks( arange(len(name)),name, rotation=80)
    plt.xlabel("Club Name")
    plt.ylabel("Articles mentioned")
    plt.suptitle('Bar chart of how many articles the club is mentioned in out of 265 articles', fontsize=8, fontweight='bold')
    plt.subplots_adjust(wspace=0.6, hspace=2, left=0.1, bottom=0.4, right=0.96, top=0.96)
    plt.savefig('task5.png')
    plt.close()
    return
    
def task6():
    import json
    import os
    import seaborn as sns
    name = []
    matchlist = []
    simlist = []
    with open(datafilepath, encoding="utf8") as f:
        Data = json.load(f)
        for clubs in Data["clubs"]:
            name.append(clubs['name'])
    totalloopnumber = 0

    csv = pd.read_csv('task5.csv')
    csv = csv["number_of_mentions"]
    outerloopnumber = 0
    data = pd.DataFrame(({"Club Names":name}))
    for club1 in name:
        innerloopnumber = 0
        simlist = []
        for club2 in name:
            clubnumber = 0
            matches = [club1, club2]
            for filename in os.listdir(articlespath):
                if filename.endswith(".txt"):
                    with open(articlespath+ '/' +filename, encoding="utf8") as f:
                        text = f.read()
                        #look in article for a match of both names
                        if all(x in text for x in matches):
                            clubnumber=clubnumber+1
            matchlist.append(clubnumber)
            #till here is correct
            both = matchlist[totalloopnumber]
            club1num = csv[outerloopnumber]
            club2num = csv[innerloopnumber]
            if(club2num+club1num==0):
                similarity=0
            else:
                similarity=(both*2)/(club1num+club2num)
            simlist.append(similarity)
            totalloopnumber = totalloopnumber+1
            innerloopnumber = innerloopnumber+1
        outerloopnumber = outerloopnumber+1
        simlistseries = pd.Series(simlist)
        data[club1] = simlistseries
    
    data = data.set_index("Club Names")
    sns.heatmap(data,cmap='viridis',xticklabels=True, cbar_kws={'label': 'Similarity Score'})
    plt.xlabel("Club Names") 
    plt.suptitle('Heat Map of similar clubs', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig('task6.png')
    plt.close()
    return
    
def task7():
    import json
    club_code = []
    goal_scored = []
    goal_conceded = []
    with open(datafilepath, encoding="utf8") as f:
        Data = json.load(f)
        for clubs in Data["clubs"]:
            club_code.append(clubs['club_code'])
            goal_scored.append(clubs['goals_scored'])
            goal_conceded.append(clubs['goals_conceded'])

        clubcode = pd.Series(club_code)
        goalscored = pd.Series(goal_scored)
        goalconceded = pd.Series(goal_conceded)
        csv = pd.DataFrame({"team_code":clubcode,"goals_scored_by_team":goalscored,"goals_scored_against_team":goalconceded})
        csv = csv.set_index('team_code')
        csv = csv.sort_values(by="team_code")
        csv.to_csv('task2mod.csv')

    import os
    from numpy import arange
    name = []
    clubnumberlist = []
    with open(datafilepath, encoding="utf8") as f:
        Data = json.load(f)
        for clubs in Data["clubs"]:
            name.append(clubs['name'])
    for club in name:
        clubnumber = 0
        for filename in os.listdir(articlespath):
            if filename.endswith(".txt"):
                with open(articlespath+ '/' +filename, encoding="utf8") as f:
                    text = f.read()
                    if(text.count(club)>0):
                        clubnumber=clubnumber+1 
                    #print(filename+str(clubnumber))
        clubnumberlist.append(clubnumber)
    clubnumber = pd.Series(clubnumberlist)
    name = pd.Series(name)
    csv = pd.DataFrame({"club_name":name,"number_of_mentions":clubnumber})
    csv = csv.sort_values(by="club_name")
    csv = csv.set_index('club_name')
    csv.to_csv('task5mod.csv')

    t2csv = pd.read_csv('task2mod.csv')
    t5csv = pd.read_csv('task5mod.csv')

    scoredlist = t2csv["goals_scored_by_team"]
    t5csv["goals_scored_by_team"]=scoredlist
    for row in t5csv.iterrows():
        plt.scatter(t5csv["goals_scored_by_team"],t5csv["number_of_mentions"],color='green')
    plt.grid(True)
    plt.xlabel("Goals scored")
    plt.ylabel("Article mentions")
    plt.suptitle('Scatterplot of goals scored and article mentions per team', fontsize=10, fontweight='bold')
    plt.savefig('task7.png')
    plt.close()
    return
    
def task8(filename):
    import re
    import nltk
    # first time:
    #nltk.download('punkt')
    #nltk.download('stopwords')
    #
    from nltk.corpus import stopwords
    with open(filename, encoding="utf8") as f:
        text = f.read()
        pattern = '[^a-zA-Z\s\n	]'
        newtext = re.sub(pattern, ' ', text) #removed special chars
        newtext = re.sub('\s\s+', ' ', newtext) #remove spacing and tab and newline
        newtext = newtext.lower() #Change all uppercase characters to lower case
        wordList = nltk.word_tokenize(newtext) #Tokenize the resulting string into words
        stopWords = set(stopwords.words('english')) #Remove all stopwords in nltk’s list of English stopwords from the resulting list
        filteredList = [w for w in wordList if not w in stopWords]
        for word in filteredList:                 #remove 1 char long words
            if(len(word)==1):
                filteredList.remove(word)
   #print(filteredList)
    return filteredList
    
def task9(): 
    import os
    import math
    from numpy import dot
    from numpy.linalg import norm
    from sklearn.feature_extraction.text import TfidfTransformer
    def cosine_sim(v1,v2):
        return dot(v1,v2)/(norm(v1)*norm(v2))
    bagofwordsdict = {}
    listoflistmatrix = []
    uniqueWords = []
    articlenames = sorted(os.listdir(articlespath))
    #get all unique words in all articles
    for filename in articlenames:
        if filename.endswith(".txt"):
            bagofwords = task8(articlespath+'/'+filename)
            bagofwordsdict[filename] = bagofwords
            uniqueWords = set(bagofwords).union(set(uniqueWords))
    
    #counting
    for filename in articlenames:
        if filename.endswith(".txt"):
            numOfWords = dict.fromkeys(uniqueWords, 0)
            for word in bagofwordsdict[filename]:
                numOfWords[word] += 1
            matrix = list(numOfWords.values())
            listoflistmatrix.append(matrix)
    
    transformer=TfidfTransformer()
    vectors = transformer.fit_transform(listoflistmatrix).toarray()
    #got vectors in 2d array
    
    i=0
    j=0
    listfile1 = []
    listfile2 = []
    listsim = []
    for i in range(0,len(articlenames)):
        #if [i].endswith(".txt"): 
            #do not want duplicates so i is start range
            for j in range(i,len(articlenames)):
                #if j.endswith(".txt"):
                    if i!=j:
                        cosim= cosine_sim(vectors[i],vectors[j])
                        filename1=articlenames[i]
                        filename2=articlenames[j]
                        listfile1.append(filename1)
                        listfile2.append(filename2)
                        listsim.append(cosim)
    file1 = pd.Series(listfile1)
    file2 = pd.Series(listfile2)
    sim = pd.Series(listsim)
    csv = pd.DataFrame({"article1":file1,"article2":file2,"similarity":sim})
    csv = csv.sort_values(by="similarity", ascending=False)
    csv = csv.head(10)
    csv = csv.set_index('article1')
    csv.to_csv('task9.csv')
    return