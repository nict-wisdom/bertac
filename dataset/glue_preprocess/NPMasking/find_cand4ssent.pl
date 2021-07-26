# Copyright (c) 2021-present, Data-driven Intelligent System Research Center (DIRECT), National Institute of Information and Communications Technology (NICT).
# All rights reserved.
use strict;
use warnings;

if($#ARGV != 0){
	print "perl find_cand.pl stem[train, dev]\n";
	exit;
}
my $stem = shift @ARGV;
unless($stem =~ /(?:CoLA|SST-2).train/ or $stem =~ /(?:CoLA|SST-2).dev/){
	print "perl find_cand.pl stem[train, dev]\n";
	exit;
}

my $debug = 0;
my %sent = ();
my %np = ();
my %pron = ();
my $ifname = "np/$stem.tag";
my $ofname = ">masked/$stem.masked";
open(FIN,$ifname) or die "Can't open the $ifname\n";
open(FOUT,$ofname) or die "Can't open the $ofname\n";
my $line = 0;
while(my $aline = <FIN>){
	chomp($aline);
	if($aline ne "END"){
		my @temp = split(/\t/,$aline);
		my $id = $temp[0];
		my $type = $temp[1];
		my $sid  = $temp[2];
		my $cont = $temp[3];
		if($type eq "TOKENS"){
			$sent{$id}{$sid} = $cont;
		}
		elsif($type eq "NP"){
			my @temp2 = split(/ /,$cont);
			my $cont_num = 0;
			my @onp = ();
			my @snp = ();
			for(my $i =0; $i <= $#temp2; $i+=2){
				my $word = $temp2[$i];
				my $pos = $temp2[$i+1];
				push @onp, $word;
				unless($pos eq "PRON" or $pos eq "DET" or 
					   $pos eq "PUNCT" or $pos eq "SYM" or 
					   $pos eq "NUM"){
					push @snp, $word;
				}
			}
			if($#temp2 == 1 and $temp2[1] eq "PRON"){
				my $pn = $temp2[0];
				$pn =~ tr/A-Z//;
			}
			if($#snp != -1){
				my $snp_str = join(" ",@snp);
				$snp_str =~ tr/A-Z/a-z/;
				my $onp_str = join(" ",@onp);
				$np{$id}{$sid}{$onp_str} = $snp_str;
			}
		}
		else{
			print "WRONG TYPE\t$type\n";
		}
	}
}

my @key = sort keys %sent;
for(my $i =0; $i <= $#key; $i++){
	my @key2 = sort keys %{$sent{$key[$i]}};
	my $sent1 =  $sent{$key[$i]}{"1"};
	my @np1 = keys %{$np{$key[$i]}{"1"}};
	if($#np1 == -1){
		print FOUT "NO_NP1\t$key[$i]\t$sent1\t[MASK] $sent1\n";
	}
	else{
		my $max_id = -1;
		my $max_wlen = -1;
		my $max_len = -1;
		for(my $j = 0; $j <= $#np1; $j++){
			my @words = split(/ /,$np{$key[$i]}{"1"}{$np1[$j]});
			my $clen = length($np{$key[$i]}{"1"}{$np1[$j]});
			if($#words > $max_wlen){
				$max_wlen = $#words;
				$max_len = $clen;
				$max_id = $j;
			}
			elsif($#words == $max_wlen){
				if($clen > $max_len){
					$max_len = $clen;
					$max_id = $j;
				}
			}
		}
		my $lcs = $np1[$max_id];
		my $idx = index($sent1, $lcs);
		if($idx == -1){
			print "Can't find $lcs in $sent1\n";
			exit;
		}
		else{
			my $msent1 = GetMaskedStr($sent1, $lcs);
			print FOUT "LCS\t$key[$i]\t$sent1\t$msent1\n";
		}
	}
}

sub GetMaskedStr{
	my ($str, $lcs) = @_;
	my $idx = index($str, $lcs);
	my @res = ();
	if($idx != 0){
		push @res, substr($str, 0, $idx);
	}
	push @res, "[MASK]";
	my $residue = length($str) - ($idx+length($lcs));
	if($residue != 0){
		push @res, substr($str, $idx+length($lcs), $residue);
	}
	my $res_str = join("",@res);
	return $res_str;
}

