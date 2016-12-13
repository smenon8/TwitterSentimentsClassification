echo "obama_bayes"
python TwitterSentiments.py -c obama -cv True -clf bayes > ../data/obama_bayes.dat

echo "romney_logisitic"
python TwitterSentiments.py -c romney -cv True -clf bayes > ../data/romney_bayes.dat

echo "obama_logisitic"
python TwitterSentiments.py -c obama -cv True -clf logistic > ../data/obama_logisitic.dat

echo "romney_dtree"
python TwitterSentiments.py -c romney -cv True -clf logistic > ../data/romney_logisitic.dat

echo "obama_dtree"
python TwitterSentiments.py -c obama -cv True -clf dtree > ../data/obama_dtree.dat

echo "romney_dtree"
python TwitterSentiments.py -c romney -cv True -clf dtree > ../data/romney_dtree.dat

echo "obama_rf"
python TwitterSentiments.py -c obama -cv True -clf rf > ../data/obama_rf.dat

echo "romney_rf"
# python TwitterSentiments.py -c romney -cv True -clf rf > ../data/romney_rf.dat

echo "obama_svm"
python TwitterSentiments.py -c obama -cv True -clf svm > ../data/obama_svm.dat

echo "romney_svm"
python TwitterSentiments.py -c romney -cv True -clf svm > ../data/romney_svm.dat

echo "obama_ada_boost"
python TwitterSentiments.py -c obama -cv True -clf ada_boost > ../data/obama_ada_boost.dat

echo "romney_ada_boost"
python TwitterSentiments.py -c romney -cv True -clf ada_boost > ../data/romney_ada_boost.dat

# Sreejith Menon
# Rajan Bhandari