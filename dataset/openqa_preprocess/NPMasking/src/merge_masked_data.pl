# Copyright (c) 2021-present, Data-driven Intelligent System Research Center (DIRECT), National Institute of Information and Communications Technology (NICT).
# # All rights reserved.

use strict;
use warnings;

if($#ARGV != 1){
	print "perl merge_masked.pl filestem[train.0...] tmp_dir\n";
	exit;
}

my $stem = shift @ARGV;
my $tmp_dir = shift @ARGV;

my $ifname1 = "$tmp_dir/cand/$stem.cand";
my %db = ();

open(FIN, $ifname1) or die "Can't open $ifname1\n";
while(my $aline = <FIN>){
	chomp($aline);
	my @temp = split(/\t/,$aline);
	unless($#temp == 7 or $#temp == 5){
		print "$#temp: $aline\n";
		exit;
	}
	my $type = $temp[0];
	my $id = $temp[1];
	my $s1 = $temp[2];
	my $s2 = $temp[3];
	my $m1 = $temp[4];
	my $m2 = $temp[5];
	my $m3 = "null";
	my $m4 = "null";
	if($#temp > 5){
		$m3 = $temp[6];
		$m4 = $temp[7];
	}
	if($type eq "LCS"){ # store only the LCS-type data
		$db{$id} = "$s1\t$s2\t$m1\t$m2\t$m3\t$m4";
	}
}
close FIN;

my $ifname2 = "$tmp_dir/sents/$stem";
my $ofname = ">$tmp_dir/masked/$stem.masked";
my $line = 0;
open(FIN, $ifname2) or die "Can't open $ifname2\n";
open(FOUT, "$ofname") or die "Can't open $ofname\n";
while(my $aline = <FIN>){
	chomp($aline);
		my @temp = split(/\t/,$aline);
		my $idx  = $temp[0];
		my $sent1 = $temp[1];
		my $sent2 = "";
		if($#temp == 1){
			push @temp, $sent2;
		}
		else{
			$sent2 = $temp[2];
		}
		push @temp, $sent1;
		push @temp, $sent2;
		$aline = join("\t",@temp); 
		if(exists($db{$idx})){
			my @masked = split(/\t/,$db{$idx});
			$temp[$#temp-1] = $masked[2];
			$temp[$#temp] = $masked[3];
			my $new_sample = join("\t",@temp);
			print FOUT "$new_sample\n";
		}
		else{
			print FOUT "$aline\n";
		}
	$line++;
}
close FIN;
close FOUT;

