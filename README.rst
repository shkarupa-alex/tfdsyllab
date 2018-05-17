tfdsyllab
=========
Word syllables prediction with deep learning


Obtain training data
--------------------
1. Obtain wiktionary dump
  ``wget https://dumps.wikimedia.org/ruwiktionary/20180501/ruwiktionary-20180501-pages-articles-multistream.xml.bz2``

2. Unpack dump
  ``bzip2 -dk ruwiktionary-20180501-pages-articles-multistream.xml.bz2``

3. Combine multiline templates
  ``cat ruwiktionary-20180501-pages-articles-multistream.xml | sed -e ':a' -e 'N' -e '$!ba' -e 's/\n[ \t]*\|/ |/g' > syllabs_0.txt``

4. Filter only lines with syllables
  ``cat syllabs_0.txt | grep 'слоги={{' > syllabs_1.txt``

5. Filter only required language
  ``cat syllabs_1.txt | grep ' ru[\| ]' > syllabs_2.txt``

6. Extract syllables from templates
  ``cat syllabs_2.txt | sed -n 's/.*слоги={*{\([^}]*\)}}*.*/\1/p' > syllabs_3.txt``

7. Make all lower
  ``cat syllabs_3.txt | tr '[:upper:]' '[:lower:]' > syllabs_4.txt``

8. Replace misspellings
  ``cat syllabs_4.txt | sed 's/по слогам/по-слогам/' > syllabs_5.txt``

9. Remove accents
  ``cat syllabs_5.txt | sed "s/̀//g" | sed "s/́//g" | sed "s/&amp;#768;//g" | sed "s/&amp;#769;//g" > syllabs_6.txt``

10. Remove unmeaning chars
  ``cat syllabs_6.txt | sed 's/«//g' | sed 's/»//g' | sed "s/'//g" > syllabs_7.txt``

11. Replace incorrect middots
  ``cat syllabs_7.txt | sed 's/&amp;middot;/|.|/g' | sed 's/·/|.|/g' | sed 's/||/|/g' | sed 's/||/|/g' > syllabs_8.txt``

12. Convert old syllables syntax
  ``cat syllabs_8.txt | grep 'слоги|' | sed 's/слоги\|/по-слогам\|/' | sed 's/\//\|/g' > syllabs_9.txt``
  ``cat syllabs_8.txt | grep -v 'слоги|' >> syllabs_9.txt``

13. Leave only lines with correct syntax
  ``cat syllabs_9.txt | grep 'по-слогам|' > syllabs_10.txt``

14. Remove rest of template
  ``cat syllabs_10.txt | sed 's/по-слогам|//g' > syllabs_11.txt``

15. Trim spaces
  ``cat syllabs_11.txt | sed 's/^[[:blank:]]*//;s/[[:blank:]]*$//' > syllabs_12.txt``

16. Remove lines with spaces
  ``cat syllabs_12.txt | grep -v ' ' > syllabs_13.txt``

17. Remove lines without requred language letters
  ``cat syllabs_13.txt | grep 'а\|б\|в\|г\|д\|е\|ё\|ж\|з\|и\|й\|к\|л\|м\|н\|о\|п\|р\|с\|т\|у\|ф\|х\|ц\|ч\|ш\|щ\|ъ\|ы\|ь\|э\|ю\|я' > syllabs_14.txt``

18. Remove incorrect lines
  ``cat syllabs_14.txt | grep -v '^[\.\-]' | grep -v '{' > syllabs_15.txt``

19. Remove duplicates
  ``sort -u syllabs_15.txt -o syllabs_16.txt``

20. Extract titles from wiktionary dump, then correct and split syllables into correct and incorrect markup
  ``cat syllabs_0.txt | grep '<title>' | grep '</title>' | sed 's/<title>//g' | sed 's/<\/title>//g' | tr '[:upper:]' '[:lower:]' | sed 's/^[[:blank:]]*//;s/[[:blank:]]*$//' > titles.txt``

21. Split into correct and incorrect markup (with default vowels from russian language)
  ``python -m tfdsyllab.correct syllabs_16.txt titles.txt syllabs_correct.txt syllabs_incorrect.txt``


As result you should get about 99К examples


Convert training data into proper format
----------------------------------------
  ``python -m tfdsyllab.convert syllabs_correct.txt ./``


Extract character vocabulary
----------------------------
  ``python -m tfdsyllab.vocab .``


Train and export model
----------------------
  ``python -m tfdsyllab.train . model -export_path export``


Use detector
------------
  ``
  from tfdsyllab.detect import SyllablesDetector
  detector = SyllablesDetector(exported_model_dir)
  result = detector.detect([u'привет', u'пока', u'японо-российский'])
  print(result)
  ``


