for f in ./*.mp4
do
    echo "file '$f'" >> mylist.txt
done
ffmpeg -safe 0 -y -f concat -i mylist.txt -c copy output.mp4;
ffmpeg -y -i output.mp4 -f gif output.gif;
rm mylist.txt
rm output.mp4
