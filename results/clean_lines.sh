#remove duplicated lines into csv
rm tmp.log &> /dev/null
head -n 1 $1 > tmp.log
i=0

while read -r line
do
	i=$i+1
	if [[ $((i%2)) -ne 0 ]]
	then
		continue
	fi
	echo $line >> tmp.log
done < $1

mv tmp.log $1
