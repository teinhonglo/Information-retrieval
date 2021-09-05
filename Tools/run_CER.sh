
ref=$1
hyp=$2

for f in $ref $hyp; do
    fout=${f%.txt}.chars.txt
      cat $f |  perl -CSDA -ane '
        {
          print $F[0];
          foreach $s (@F[1..$#F]) {
            if (($s =~ /\[.*\]/) || ($s =~ /\<.*\>/) || ($s =~ "!SIL")) {
              print " $s";
            } else {
              @chars = split "", $s;
              foreach $c (@chars) {
                print " $c";
              }
            }
          }
          print "\n";
        }' > $fout
done

ref=${ref%.txt}.chars.txt
hyp=${hyp%.txt}.chars.txt

python WERv2.py $ref $hyp
