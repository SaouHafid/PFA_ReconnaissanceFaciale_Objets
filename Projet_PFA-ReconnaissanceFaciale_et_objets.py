from tkinter import *
from tkinter import ttk

#Définition de la fonction de détection et création de visages, et insertion de données dans la base de données.
def functionDetection():
    print('Détection et création de visages et insertion de données')

    import cv2
    import numpy as np
    import sqlite3
    import os

    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cam = cv2.VideoCapture(0);

    def InsertUpdate(Id,Nom,Age,Sexe,Classe):
        BASE_DIR = os.path.dirname(os.path.abspath(r'C:\Users\LENOVO\AppData\Local\Programs\Python\Python36-32\ReconnaissanceFaciale\Projet_PFA.py'))
        dbPath = os.path.join(BASE_DIR, "FaceBase.db")
        conn = sqlite3.connect(dbPath) 
        cmd = "SELECT * FROM personne WHERE Id="+str(Id) 
        cursor = conn.execute(cmd)
        #isCasierJudicExist = 0 
        #for row in cursor:
        #    isCasierJudicExist = 1
        #if(isCasierJudicExist == 1):
        #    cmd = "UPDATE personne SET Nom"+str(Nom)+"WHERE Id="+str(Id)
        #else:
        cmd = "INSERT INTO personne(Id,Nom,Age,Sexe,Classe) VALUES ("+str(Id)+","+str(Nom)+","+str(Age)+","+str(Sexe)+","+str(Classe)+")"
        conn.execute(cmd)
        conn.commit()
        conn.close() 

    id = input("Entrez votre idUtilisateur: ")
    nom = input("Entrez votre nom: ")
    age = input("Entrez votre âge: ")
    sexe = input("Entrez votre sexe: ")
    classe = input("Entrez votre classe: ")
    InsertUpdate(id,nom,age,sexe,classe)
    sampleNum = 0
    while True:
        ret,img = cam.read()
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray,1.3,5)
        for(x,y,w,h) in faces:
            sampleNum =  sampleNum + 1
            cv2.imwrite("dataSet/Utilisateur."+str(id)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
            cv2.rectangle(img, (x,y),(x+w,y+h),(0,0,225),2)
            cv2.waitKey(100)
        cv2.imshow("Face",img)
        cv2.waitKey(100)
        if sampleNum>20:
            cam.release()
            cv2.destroyAllWindows()
            break 

#définition de la méthode de quittage de l'application desktop.     
def quitter():
    import tkinter.messagebox
    answer = tkinter.messagebox.askquestion('Question1', "Voulez-vraiment quitter l'application DetectMe?")
    print(answer) 
    print('quitter')
    
#Définition de la méthode de la reconnaissance d'un visage d'une personne, et de l'affichage de ses données:
def reco():
    print('Reconnaissance faciale')

    import numpy as np
    import cv2 
    import os
    import pickle 
    from PIL import Image
    import sqlite3
    from sqlite3 import OperationalError
      
    #recognizer = cv2.createLBPHFaceRecognizer()
    recognizer = cv2.face.createLBPHFaceRecognizer()
    recognizer.load("recognizer/trainningData.yml")
    faceDetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    path = 'dataSet'

    def getProfile(id):
        BASE_DIR = os.path.dirname(os.path.abspath(r'C:\Users\LENOVO\AppData\Local\Programs\Python\Python36-32\ReconnaissanceFaciale\detector.py'))
        db_Path = os.path.join(BASE_DIR, "FaceBase.db")
        conn = sqlite3.connect(db_Path) 
        cmd = "SELECT * FROM personne WHERE Id="+str(id);
        #"""SELECT name, age FROM users""" 

        cursor = conn.execute(cmd)
        profile = None
        for row in cursor:
            profile = row 
        conn.close() 
        return profile

    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX #Create font
    fontScale = 1.4
    fontColor = (0, 255, 0)
    while True: 
        ret,img = cam.read()
        #locy = int(img.shape[0]/2) # the text location will be in the middle
        #locx = int(img.shape[1]/2)
        # print ret: It's just for testing if there is some issue with camera and then changing
        # cam = cv2.VideoCapture(0) to cam = cv2.VideoCapture(1) et vis-versa
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray,1.3,5)
        for(x,y,w,h) in faces:
            
            id, conf = recognizer.predict(gray[y:y+h,x:x+w])
            cv2.rectangle(img, (x,y),(x+w,y+h),(255,0,0), 2)
            profile = getProfile(id)
            if(profile!=None): 
                cv2.putText(img, str(profile[1]), (x,y+h+30), font, fontScale, fontColor)
                cv2.putText(img, str(profile[2]), (x,y+h+60), font, fontScale, fontColor) 
                cv2.putText(img, str(profile[3]), (x,y+h+90), font, fontScale, fontColor)
                cv2.putText(img, str(profile[4]), (x,y+h+120), font, fontScale, fontColor)
                 
        cv2.imshow("Face",img)
        cv2.waitKey(10)
	


#Définition de la méthode de la détection, et de pointillage sur un objet réel :    	
def objDetect():
    print('Detection dobjets')
    import cv2
    import numpy as np

    MIN_MATCH_COUNT=30

    detector=cv2.xfeatures2d.SIFT_create()
    FLANN_INDEX_KDITREE=0 
    flannParam=dict(algorithm=FLANN_INDEX_KDITREE,tree=5)
    flann=cv2.FlannBasedMatcher(flannParam,{})

    trainImg=cv2.imread("trainingData/fixPhone.jpg",0)
    trainKP,trainDesc=detector.detectAndCompute(trainImg,None)

    cam=cv2.VideoCapture(0)
    while True:
        ret, QueryImgBGR=cam.read()
        QueryImg=cv2.cvtColor(QueryImgBGR,cv2.COLOR_BGR2GRAY)
        queryKP,queryDesc=detector.detectAndCompute(QueryImg,None)
        matches=flann.knnMatch(queryDesc,trainDesc,k=2)

        goodMatch=[]
        for m,n in matches:
            if(m.distance<0.75*n.distance):
                goodMatch.append(m)

        if(len(goodMatch)>MIN_MATCH_COUNT):
            tp=[]
            qp=[]
            for m in goodMatch:
                tp.append(trainKP[m.trainIdx].pt)
                qp.append(queryKP[m.queryIdx].pt)
            tp,qp = np.float32((tp,qp))
            H,status=cv2.findHomography(tp,qp,cv2.RANSAC,3.0)
            h,w=trainImg.shape
            trainBorder=np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
            queryBorder=cv2.perspectiveTransform(trainBorder,H)
            cv2.polylines(QueryImgBGR,[np.int32(queryBorder)],True,(0,255,0),5)

        else:
            print ("Matches insuffisants trouves - %d/%d"%(len(goodMatch),MIN_MATCH_COUNT))
        cv2.imshow('result',QueryImgBGR)
        cv2.waitKey(10)


def trainer():
    print('trainer')
    import os
    import cv2
    import numpy as np
    from PIL import Image

    recognizer=cv2.face.createLBPHFaceRecognizer()
    path =r'C:\Users\LENOVO\AppData\Local\Programs\Python\Python36-32\ReconnaissanceFaciale\dataSet'

    def getImageWithID(path): 
        imagePaths = [os.path.join(path,f)for f in os.listdir(path)]
        faces = [] 
        IDs = [] 
        for imagePath in imagePaths:
            faceImg = Image.open(imagePath).convert('L')
            faceNp = np.array(faceImg,'uint8')
            ID = int(os.path.split(imagePath)[-1].split('.')[1])
            faces.append(faceNp)
            print(ID)
            IDs.append(ID)
            cv2.imshow("trainning",faceNp)
            cv2.waitKey(10)
        return IDs, faces

    Ids,faces = getImageWithID(path) 
    recognizer.train(faces, np.array(Ids)) 
    recognizer.save('recognizer/trainningData.yml')
    #fichier qui va aider l'algorithme de l'application à identifier les images numérotées crées dans la base de données d'images, pour l'aider à la reconnaissance.  
    cv2.destroyAllWindows()

#Defintion de la methode apropos   
def apropos():
    print('A propos!')
    import tkinter.messagebox
    #Creation de message contenat le résumé de l'app DetectMe:
    tkinter.messagebox.showinfo("A propos de l'application DetectMe","DetectMe est une application constitue le fruit de travail accompli dans le cadre du Projet de Fin d’Année au sein de l’Ecole Nationale des Sciences Appliquées d’Al Hoceima. Notre projet ayant pour objectif de détection et reconnaissance faciale ainsi que les objets qu’il substitue de l’analyse du contenu des images et la reconnaissance de formes qui sont des domaines d’application en pleine expansion de nos jours, notamment grâce à l’efficacité offerte par la puissance des machines. La reconnaissance faciale est l’une des tendances de sujets de recherches les plus étudiées. En effets, elle correspond à ce que les humains utilisent par l’interaction visuelle, ainsi l'efficacité de la reconnaissance faciale dépend d’une façon primordiale de la qualité de l'image d’ou le système tente-t-il de faire la distinction entre des sujets coopérants et non coopérants, et aussi d’Algorithmes d'identification qui est le facteur de performance clé, et la puissance des algorithmes utilisés pour déterminer des similitudes entre les traits des visages.  Et une Base de données fiables d’ou elle dépend de la taille et de la qualité des bases de données utilisées : pour reconnaître un visage, on doit pouvoir le comparer à un autre. Et la difficulté s’établit sur des points de correspondance entre la nouvelle image et l'image source, en d'autres termes, les photos d'individus connus. En conséquence, plus la base de données d'images ciblées est importante, plus il est possible d'y trouver des correspondances. Le développement étant amorcé, le test et la qualification de la solution ont été réalisés pour assurer le bon fonctionnement de l’application. Un ensemble de technologies d'outils ont été utilisés pour ce projet : Python, (OpenCv, numpy, Matpoltlib), SQLite, Tk Interface, PIL .. ainsi que d’autres dépendances.")

#Definition de la metohde guide:
def guide():
    print('guide')
    
    import tkinter.messagebox
    tkinter.messagebox.showinfo("Le guide pour l'application DetectMe","Pour bien s'adapter avec l'application DetectMe, nous vous présontons le guide qui suit: - En cliquant sur le boutton 'Détecter un visage, et enregistrer ses données', vous allez détecter par la caméra de l'ordinateur une personne debout devant le PC, puis vous allez enregistrer ses données personnelles. - En cliquanr sur le boutton 'Créer l'identifiant de visages' vous allez aider DetectMe à identifier la personne à reconnaître par la suite. - En cliquanr sur le boutton 'Reconnaître un visage, et afficher ses données', vous allez afficher les informations de la personne détectée par la caméra. - En cliquant sur le boutton 'Détecter un objet réel', vous allez profiter d'une détection d'un objet réel exposé devant la caméra de l'ordianteur.")
    
#Adoption de la bibliothèque Tk Interface (tkinter) pour créer l'interface graphique de l'application:
root = Tk()
root.title("Application de reconaissance faciale et détection d'objets - DetectMe")
root.configure(background='#D1B606')
 
#Définition du menu:           
menu = Menu(root)
root.config(menu=menu)

subMenu = Menu(menu)

menu.add_cascade(label="Edit", menu=subMenu)
subMenu.add_separator()
subMenu.add_command(label="Quitter", command=quitter)

subMenu2 = Menu(menu)
menu.add_cascade(label="Manipulation de visage", menu=subMenu2)
subMenu2.add_command(label="Détection d'un visage", command=functionDetection)
subMenu2.add_command(label="Reconnaissance d'un visage", command=reco)
subMenu2.add_command(label="Création de l'identifiant de visages", command=trainer)


subMenu3 = Menu(menu)
menu.add_cascade(label="Manipulation d'objet", menu=subMenu3)
subMenu3.add_command(label="Détection d'un objet",command=objDetect)


aideMenu = Menu(menu)
menu.add_cascade(label="Aide", menu=aideMenu)
aideMenu.add_command(label="A propos?", command=apropos)
subMenu.add_separator()
aideMenu.add_command(label="Guide", command=guide)

  #Importation d'images
DetectFace=PhotoImage(file=r'DetectMe.png') 
resize_detectme=DetectFace.subsample(4,3)
KnowFace=PhotoImage(file=r'KonwMe.PNG')
resize_konowme=KnowFace.subsample(3,4)
DetectObjet=PhotoImage(file=r'detectObjet.png')
resize_detectObjet=DetectObjet.subsample(3,3)
Trainer=PhotoImage(file=r'trainMe.png')
resize_trainer=Trainer.subsample(3,3)
    
label = Label(root, text="Cliquez-vous sur un bouton au-dessus pour faire un traitement graphique bien précis:",font=('Arial',12,'bold'),compound=RIGHT, bg="#FCFDFD")
label.pack(padx=1,pady=4)

button1 = ttk.Button(root, text="Détecter un visage, et enregistrer ses données")
button1.pack(padx = 8, pady = 8)
button1.config(command=functionDetection)

button4 = ttk.Button(root, text="Créer l'identifiant de visages") 
button4.pack(padx = 8, pady = 8)
button4.config(command=trainer)

button2 = ttk.Button(root, text="Reconnaître un visage, et afficher ses données")
button2.pack(padx = 8, pady = 8)
button2.config(command=reco)

button3 = ttk.Button(root, text="Détecter un objet réel") 
button3.pack(padx = 8, pady = 8)
button3.config(command=objDetect)


   #Concernant les styles des buttons:
style = ttk.Style()
style.theme_use('classic')
style.configure('Info.TButton',font=('Arial',20,'bold'))
button3.configure(style='Info.TButton')
button2.configure(style='Info.TButton') 
button1.configure(style='Info.TButton')
button4.configure(style='Info.TButton') 

#le background de l'application:
#background_image = PhotoImage(file=r'DetectMe.PNG')
#background_label = ttk.Label(image=background_image)
#background_label.place(x=0, y=0, relwidth=1, relheight=1)
#background_label.image = background_image


button1.config(image=resize_detectme, compound=LEFT)
button2.config(image=resize_konowme, compound=LEFT)
button3.config(image=resize_detectObjet, compound=LEFT)
button4.config(image=resize_trainer, compound=LEFT)  

#La barre d'informations facultative:
status = Label(root, text="Application réalisée par les deux étudiants : Soufiane TAZI && Abdelhafid SAOU dans le cadre de projet PFA ©.",bd=1,relief=SUNKEN,anchor=E)
status.pack(side=BOTTOM)

#c = Canvas(root,width=500, height=500)
#c.pack()
# image
#fond = PhotoImage(file=r"F:\s4\GP\fbi-reconnaissance-faciale.jpg")
#image de fond 
#c.create_image(0, 0, image=fond) 
