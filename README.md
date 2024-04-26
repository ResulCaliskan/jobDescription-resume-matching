# jobDescription-resume-matching


NLP modelini indirmek için gereken kod 
```
python -m spacy download en_core_web_sm
```
Terminal ile indirmek isterseniz:
```
!python -m spacy download en_core_web_sm
```


Spacy kütüphanesini indirirken ERROR: Could not install packages due to an OSError: [WinError 2] Sistem belirtilen dosyayı bulamıyor: 'C:\\Python312\\Scripts\\f2py.exe' -> 'C:\\Python312\\Scripts\\f2py.exe.deleteme'
böyle bir hata ile karşılaşırsanız
```
pip install spacy --user
```
kodunu kullanarak bu hatayı çözebilirsiniz
