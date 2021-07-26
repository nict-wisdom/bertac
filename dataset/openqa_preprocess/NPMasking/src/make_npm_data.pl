# Copyright (c) 2021-present, Data-driven Intelligent System Research Center (DIRECT), National Institute of Information and Communications Technology (NICT).
# All rights reserved.

use strict;
use warnings;
use Encode;
use utf8;

binmode STDOUT, ":encoding(utf8)";

if($#ARGV != 2){
	print "perl make_npm_data.pl set_stem[searchqa|quasart] dstem[train|dev|test] tmp_dir\n";
	exit;
}

my $stem = shift @ARGV;
my $dstem = shift @ARGV;
my $tmp_dir = shift @ARGV;

unless($stem eq "searchqa" or $stem eq "quasart"){
	print "perl make_npm_data.pl set_stem[searchqa|quasart] dstem[train|dev|test]\n";
	exit;
}

unless($dstem eq "train" or $dstem eq "dev" or $dstem eq "test"){
	print "perl make_npm_data.pl set_stem[searchqa|quasart] dstem[train|dev|test]\n";
	exit;
}

my %db = ();
open(FIN,"<:encoding(utf8)","$tmp_dir/$stem.$dstem.all.masked.tsv") or die "Can't open the file $tmp_dir/$stem.$dstem.all.masked.tsv\n";
while(my $aline = <FIN>){
	chomp($aline);
	my @temp = split(/\t/,$aline);
	my @id = split(":::",$temp[0]);
	my $cid = "$id[0]\t$id[1]";
	$db{$cid} = "$temp[3]\t$temp[4]";
}
close FIN;

my @qdb = ();
open(FIN,"<:encoding(utf8)","$stem/$dstem.q") or die "Can't open the file $stem/$dstem.q\n";
while(my $aline = <FIN>){
	chomp($aline);
	push @qdb, $aline;
}
close FIN;

open(FIN,"<:encoding(utf8)","$stem/$dstem.json.txt") or die "Can't open the file $stem/$dstem.json.txt\n";
open(FOUT,">:encoding(utf8)","$stem/$dstem.npm") or die "Can't open the file\n";
my $pid = "none";
my %tmp = ();
my $qidx = 0;
while(my $aline = <FIN>){
	chomp($aline);
	my @temp = split(/\t/,$aline);
	my @id = split(":::",$temp[0]);
	if($pid ne "none" and $id[0] ne $pid){
		my @key = sort {$a<=>$b} keys %{$tmp{$pid}};
		if($#key == -1){
			print "No answer for $pid\n";
			exit;
		}
		my @cont = split(/\t/,$tmp{$pid}{$key[0]});
		my $aq = ConvertQ($qdb[$qidx]);
		my $pq = ConvertQ($cont[2]);
		if($aq ne $pq){
			print "Questions were different: $qidx\t$pid\n";
			print "AnswerSet: $aq\t$qdb[$qidx]\n";
			print "Processed: $pq\t$cont[2]\n";
			exit;
		}
		for(my $i =0; $i <= $#key; $i++){
			print FOUT "$tmp{$pid}{$key[$i]}\n";
		}
		print FOUT "END\n";
		%tmp = ();
		$qidx++;
	}
	my $cid = "$id[0]\t$id[1]";
	if(exists($db{$cid})){ #
	unless(exists($db{$cid})){
		print "Can't find the npm for $cid\n";
		exit;
	}
	my @tempk = split(/ /,$temp[2]);
	if($#tempk >= 5){
		$tmp{$id[0]}{$id[1]} = "$cid\t$temp[1]\t$temp[2]\t$db{$cid}";
	}
	} #
	$pid = $id[0];
}

my @key = sort {$a<=>$b} keys %{$tmp{$pid}};
if($#key == -1){
	print "No answer for $pid\tfinal\n";
	exit;
}
my @cont = split(/\t/,$tmp{$pid}{$key[0]});
if($qdb[$qidx] ne $cont[2]){
	print "Questions were different: $qidx\t$pid\n";
	print "AnswerSet: $qdb[$qidx]\n";
	print "Processed: $cont[2]\n";
}
for(my $i =0; $i <= $#key; $i++){
	print FOUT "$tmp{$pid}{$key[$i]}\n";
}
print FOUT "END\n";

print "ProcessedQ: $qidx\t$#qdb\n";
close FIN;
close FOUT;

sub ConvertQ{
	my ($qstr) = @_;
	$qstr =~ s/-LRB-/(/g;
	$qstr =~ s/-RRB-/)/g;
	$qstr =~ s/[ ]+//g;
	$qstr =~ s/[`']+//g;
	return $qstr;
}
