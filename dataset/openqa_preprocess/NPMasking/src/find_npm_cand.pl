# Copyright (c) 2021-present, Data-driven Intelligent System Research Center (DIRECT), National Institute of Information and Communications Technology (NICT).
# All rights reserved.
use strict;
use warnings;

if($#ARGV != 1){
	print "perl find_cand.pl stem[searchqa.train.0] tmp_dir\n";
	exit;
}
my $stem = shift @ARGV;
my $tmp_dir = shift @ARGV;

my $idir = "$tmp_dir/np";
my $odir = "$tmp_dir/cand";

unless(-e $idir){
	print "input data dir ($idir) is not existing\n";
	exit;

}
unless(-e $odir){
	print "output dir ($odir) is not existing\n";
	exit;
}

my $debug = 0;
my %sent = ();
my %np = ();
my $ifname = "$idir/$stem.np";
my $ofname = ">$odir/$stem.cand";
open(FIN,$ifname) or die "Can't open the $ifname\n";
open(FOUT,$ofname) or die "Can't open the $ofname\n";
my $line = 0;
while(my $aline = <FIN>){
	chomp($aline);
	$line++;
	if($aline ne "END"){
		my @temp = split(/\t/,$aline);
		my $id = $temp[0];
		my $type = $temp[1];
		my $sid  = $temp[2];
		my $cont = $temp[3];
		if($#temp >= 3){ 
			if($type eq "TOKENS"){
				$cont =~ s/[ ]+/ /g;
				$sent{$id}{$sid} = $cont;
			}
			elsif($type eq "NP"){
				$cont =~ s/[ ]+/ /g;
				my @temp2 = split(/ /,$cont);
				my $cont_num = 0;
				my @onp = ();
				my @snp = ();
				for(my $i =0; $i <= $#temp2; $i+=2){
					my $word = $temp2[$i];
					my $pos = $temp2[$i+1];
					push @onp, $word;
					push @snp, $word;
				}
				if($#snp != -1){
					my $snp_str = join(" ",@snp);
					$snp_str =~ tr/A-Z/a-z/;
					my $onp_str = join(" ",@onp);
					$np{$id}{$sid}{$onp_str} = $snp_str;
				}
			}
		}
		else{
			if($type eq "TOKENS"){
				$sent{$id}{$sid} = "";
				#print "$line: $aline\n";
			}
		}
	}
}

my @key = sort keys %sent;
for(my $i =0; $i <= $#key; $i++){
	my @key2 = sort keys %{$sent{$key[$i]}};
	my $sent1 =  $sent{$key[$i]}{"1"};
	my $sent2 =  $sent{$key[$i]}{"2"};
	my @np1 = keys %{$np{$key[$i]}{"1"}};
	my @np2 = keys %{$np{$key[$i]}{"2"}};
	if($#np1 == -1 or $#np2 == -1){
		print FOUT "NO_NP1\t$key[$i]\t$sent1\t$sent2\t[MASK] $sent1\t[MASK] $sent2\n";
	}
	else{
		my $lcs = CheckLCS($key[$i]);
		if($lcs eq "null"){
			print FOUT "NO_NP2\t$key[$i]\t$sent1\t$sent2\t[MASK] $sent1\t[MASK] $sent2\n";
		}
		else{
			my $msent1 = GetMaskedStr($sent1, $lcs);
			my $msent2 = GetMaskedStr($sent2, $lcs);
			print FOUT "LCS\t$key[$i]\t$sent1\t$sent2\t$msent1\t$msent2\n";
		}
	}
}

sub GetMaskedStr{
	my ($str, $lcs) = @_;
	my @temp1 = split(/ /,$lcs);
	my $lstr = $str;
	$lstr =~ tr/A-Z/a-z/;
	my @cand = ();
	my @temp2 = split(/ /,$lstr);
	my @tempo = split(/ /,$str);
	for(my $j =0; $j <= $#temp2; $j++){
		if($temp1[0] eq $temp2[$j]){
			if($#temp1 == 0){
				push @cand, $j;
			}
			else{
				my $idx1 = 1;
				my $idx2 = $j+1;
				my @m1 = ();
				push @m1, $temp1[0];
				my $match = 1;
				while($idx1 <= $#temp1 and $idx2 <= $#temp2 and $match){
					if($temp1[$idx1] eq $temp2[$idx2]){
						$match = 1;
						push @m1, $temp1[$idx1];
						$idx1++;
						$idx2++;
					}
					else{
						$match = 0;
					}
				}
				if($idx1 == $#temp1+1){
					my $mstr = join(" ",@m1);
					if($mstr eq $lcs){
						push @cand, $j;
					}
				}
			}
		}
	}
	if($#cand == -1){
		print "Can't find the str\n" if $debug;
		print "LCS: $lcs\n" if $debug;
		print "SENT: $str\n" if $debug;
		exit;
	}
	my $idx = $cand[0];
	$tempo[$idx] = "[MASK]";
	for(my $i = $idx+1; $i <= $idx+$#temp1; $i++){
		$tempo[$i] = "";
	}
	my $mstr = join(" ",@tempo);
	$mstr =~ s/[ ]+/ /g;
	return $mstr;
}

sub LongestMatch{
	my ($s1, $s2) = @_;
	my @temp1 = split(/ /,$s1);
	my @temp2 = split(/ /,$s2);
	my %matched = ();
	for(my $i =0; $i <= $#temp1; $i++){
		for(my $j =0; $j <= $#temp2; $j++){
			if($temp1[$i] eq $temp2[$j]){
				$matched{$temp1[$i]} = 0 + 0.0001*length($temp1[$i]);
				my $idx1 = $i+1;
				my $idx2 = $j+1;
				my @m1 = ();
				push @m1, $temp1[$i];
				my $match = 1;
				while($idx1 <= $#temp1 and $idx2 <= $#temp2 and $match){
					if($temp1[$idx1] eq $temp2[$idx2]){
						$match = 1;
						push @m1, $temp1[$idx1];
						$idx1++;
						$idx2++;
					}
					else{
						$match = 0;
					}
				}
				if($#m1 > 0){
					my $mstr = join(" ",@m1);
					$matched{$mstr} = $#m1 + 0.0001*length($mstr); # 
				}
			}
		}
	}
	my @temp3 = sort {$matched{$b} <=> $matched{$a}} keys %matched;
	if($#temp3 != -1){
		return ($temp3[0], $matched{$temp3[0]});
	}
	return ("null", 0);
}

sub CheckLCS{
	my ($id) = @_;
	my @np1 = sort keys %{$np{$id}{"1"}};
	my @np2 = sort keys %{$np{$id}{"2"}};
	my %tmp_lcs = ();
	for(my $i =0; $i <= $#np1; $i++){
		my $snp1 = $np{$id}{"1"}{$np1[$i]};
		my $max_lcs = "";
		my $max_score = -1;
		for(my $j =0; $j <= $#np2; $j++){
			my $snp2 = $np{$id}{"2"}{$np2[$j]};
			my ($longest, $lscore) = LongestMatch($snp1, $snp2);
			if($longest ne "null"){
				if($max_score < $lscore){
					$max_lcs = $longest;
					$max_score = $lscore;
				}
			}
		}
		if($max_score != -1){
			$tmp_lcs{$max_lcs} = $max_score;
		}
	}
	my @key = sort {$tmp_lcs{$b}<=>$tmp_lcs{$a}} keys %tmp_lcs;
	my $lcs = "null";
	if($#key != -1){
		$lcs = $key[0];
	}
	return $lcs;
}

