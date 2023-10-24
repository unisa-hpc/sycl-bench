impl=$1
if [ $# -ne 1 ]
then
	echo "No implementation provided"
	exit
fi
rm -f accessor_latency_$impl.csv &> /dev/null
../build/usm_accessors_latency --device=gpu --num-runs=5 --num-launches=50000 --output=accessor_latency_$impl.csv
bash clean_lines.sh accessor_latency_$impl.csv
rm -f accessor_latency_${impl}_1rep.csv &> /dev/null
../build/usm_accessors_latency --device=gpu --num-runs=5 --num-launches=1 --output=accessor_latency_${impl}_1rep.csv
bash clean_lines.sh accessor_latency_${impl}_1rep.csv
