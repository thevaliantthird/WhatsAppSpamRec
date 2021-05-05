import os

class Text (object) :

    def __init__ (self, fileName):
        self.FileRead = open(fileName, "r",encoding = "unicode-escape")
        self.Text = []
        self.Counter = 0
        self.ifEnd = False



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


myText = Text('test.txt')
myText.ReadAndStore()
myText.Print()
