Nuclear Design and Technology - project work - Design of a Reactor Pressure Vessel

For any external user: this is not a real ReadMe file, since this code is only designed for a university project exercise, so no explaination about the code will be given.
The aim of this file is simply to teach to my collagues how to use github's repositories in order to collaborate in the most efficient way possible
:)

####################################################################
Ciao Cari
Vi mostro step by step cosa dovete fare per utilizzare una repository
###################
CAPITOLI
1- prima confiigurazione
2- come funzionano i branch
3- come modificare i file


###################
CAPITOLO 1 - PRIMA CONFIGURAZIONE
###################

se siete qui avete già un account github, ottimo
> mandatemi il vostro nome utente, io vi inviterò ad essere collaboratori. Dovete accettare l'invito dalla mail che riceverete.
> installate git per il vostro sistema operativo da
                                        https://git-scm.com/download
> scaricate le estensioni nell'IDE che state usando
                                        Github Pull Request and Issues
                                        Python
> a questo punto aspettate che io vi accetti come collaboratori prima del prossimo step
> clonate la repository da terminale
                                        git clone https://github.com/pypp1/NucDes.git

una volta fatto questo avrete scaricato la cartella (NucDes) del progetto sul vostro computer
all'interno ci sarà una struttura del tipo
                                        NucDes/
                                          └── src/
                                              └── main.py
                                      # costruzione struttura ancora in corso
> spostatevi nella cartella
                                        cd NucDes

###################
CAPITOLO 2 - COME FUNZIONANO I BRANCH
###################

Questa parte è abbastanza importante per capire la logica del tutto
Voi al momento avete salvato in locale una copia della cartella
su github c'è tutto il file aggiornato
voi quando modificate i file li modificate in locale sul vostro computer, per poi caricarli sulla cartella condivisa.
A questo punto sulla cartella condivisa ci saranno due linee temporali di codici:
      - il main, ovvero il codice effettivo
      - il branch da voi creato, ovvero il codice modificato
una volta che le modifiche sono state approvate si può fare il merge delle due linee, il che significa che il main viene sostituito dalle vostre modifiche
Questo processo è comodo perchè il sistema tiene in memoria tutte le modifiche, quindi ogni volta che c'è bisogno si può tornare indietro alla versioni precedenti.
Gli step da compiere sono quindi
modificare il codice > salvare > fare il commit delle modifiche (locale) > pushare le modifiche (condiviso) > aprire una richiesta per approvare le modifiche (pull request)
> approvare le modifiche > fare il merge delle linee

###################
CAPITOLO 3 - COME MODIFICARE I FILE
###################

  (nel terminale/command prompt)
> prima cosa bisogna aggiornare la versione che avete salvata in locale con l'ultima approvata:
                                        git pull
> ora importantissimo: non dovete modificare il main ma creare un nuovo branch e modificare quello. Per fare ciò
                                        git checkout -m nome-del-branch                             
> modifica del file
                                        boh modificatelo che vi devo dire
> salvare
                                        salva(?)
> salvare nella storia del progetto
                                        git add .
> commit
                                        git commit -m "commento riguardo quello che avete fatto"
> push su github
                                        git push -u origin nome-del-branch
>       importante: per fare il push vi chiederà username e password.
                   username è quello di github
                   per la password dovete creare un token con durata di tempo lmitata (usate 60gg così non lo dovete più fare fino a fine progetto)
                               per crearlo: github.com > settings > token > new token (classico, scopes: repo e basta è sufficiente) > copiate il token e incollatelo quando vi chiede la password
                                           occhio che quando mettete le password su terminale non vedete nessun carattere comparire, ma la avete messa, fidatevi. fate invio e bom.
> aprire una pull-request
  può essere fatta in due modi: tramite interfaccia grafica (più semplice ma un po' più sbatti) o tramite command line
    interfaccia grafica:
        apri github.com, entra nella repository > pull-reques > new > compare (selezionate il branch che volete comparare) > scrivete una descrizione di quello che avete fatto > mandate in revisione
    command line:
       brew install gh
       gh auth login          usate il protocollo http
       gh pr create --base nome_del_branch_di_origine(nel nostro caso main) --head nome_del_branch_che_avete_creato --title "titolo pr" --body "descrizione"

>  processo di review
    interfaccia grafica:
        boh rega sempre da sito così' vedete bene anche cosa sta venendo modificato, si può fare anche da command line ma non fatelo
    IMPORTANTE: SE VI SEGNALA CONFLITTI CHIEDETE A ME(LORE) O AD ALE
> merge delle linee
                                        git merge main

########### BELLA A TUTTI, GASSS ###########

              
        
