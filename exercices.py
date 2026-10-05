def compte_mots(texte):
    h={}
    L=texte.split()
    for i in range(len(L)):
        if h.get(L[i],0)==0:
            h[L[i]]=1
        else :
            h[L[i]]=h[L[i]]+1
    return h
print(compte_mots("le chat et le chien et le rat"))