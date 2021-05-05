import os
import numpy as np

class Text (object) :

    def __init__ (self, fileName):
        self.FileRead = open(fileName, "r",encoding = "charmap")
        self.Text = []



    def IfStart (self) :

        K = self.FileRead.tell()
        readUp = self.FileRead.read(20)
        res = False
        Name = ""
        if len(readUp) == 20 and readUp[2] == '/' and readUp[5] == '/' and readUp[10] == ','  \
        and readUp[11] == ' ' and readUp[14] == ':' and readUp[17] == ' ' and readUp[18] == '-' \
        and readUp[19] == ' ' and readUp[0:2].isnumeric() and readUp[3:5].isnumeric() and readUp[6:10].isnumeric() \
        and readUp[12:14].isnumeric() and readUp[15:17].isnumeric() :
            res  = True
            str = self.FileRead.read(1)
            while str[0] != ':' :
                readUp+=str
                str = self.FileRead.read(1)
            Name = readUp

        else:
            self.FileRead.seek(K)

        return (res,Name)

    def ReadAndStore (self):
        ifPrevTerm  = True
        Start = True
        Sender = ""
        Msg = ""
        while len(self.FileRead.read(1)) != 0:
            self.FileRead.seek(self.FileRead.tell()-1)
            if ifPrevTerm:
                #print(self.FileRead.tell())
                resu = self.IfStart()
                if resu[0]:
                    if Start:
                        Sender = resu[1]
                        Start = False
                    else:
                         self.Text.append((Sender,Msg))
                    #     self.Print()
                         Msg = ""
                         Sender = resu[1]
                else:
                    if len(Sender) == 0:
                        print("There is some inconsistency with the text!")
                        break
                    Msg+=self.FileRead.read(1)
            else:
                buff = self.FileRead.read(1)
                if buff == '\n':
                    ifPrevTerm = True
                Msg+=buff
        if len(Sender) != 0:
            self.Text.append((Sender,Msg))


    def Print(self):
        for (a,b) in self.Text:
            print(a+b)

    def ReturnText(self) :
        return self.Text




logs = open("log.txt","a")
fi = input("Please Enter the File Name?")
my = Text(fi)
my.ReadAndStore()
Y = my.ReturnText()

with open('pos.npy','rb') as f:
    pos = np.load(f)

with open('neg.npy','rb') as g:
    neg = np.load(g)

posL = []
negL = []
log = ""
for (a,b) in Y:
    print(b.encode('utf-8'))
    t = input("Spam(S)/Important(I)?")
    if t=='S':
        negL.append(b)
    elif t=='X':
        log+="The File "+fi+" has been done till "+a+b+"\n"
        break
    elif t=='A':
        x = 5
    else:
        posL.append(b)

logs.write(log)
logs.close()
posL = np.array(posL)
posL = np.expand_dims(posL,axis = 1)

negL = np.array(negL)
negL = np.expand_dims(negL,axis = 1)

pos = np.vstack((pos,posL))
neg = np.vstack((neg,negL))

os.remove("pos.npy")
os.remove("neg.npy")

np.save("pos.npy",pos)
np.save("neg.npy",neg)
