p=model
sources=test
files=$(ls $p)
for filename in $files
do
name=$(echo $filename| grep "[0-9]")
echo "evaluation results on ${name}.pt"
python translate.py -model ./model/$filename -src ./data/src-$sources.txt -tgt ./data/tgt-$sources.txt -output ./predict/predict.txt -gpu 0
cp -f ./predict/predict.txt ./evalTool/
cd ./evalTool
bash evaluate.sh 
cd ../
done
