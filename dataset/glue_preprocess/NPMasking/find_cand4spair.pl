# Copyright (c) 2021-present, Data-driven Intelligent System Research Center (DIRECT), National Institute of Information and Communications Technology (NICT).
# All rights reserved.
use strict;
use warnings;
use LCSS;

if($#ARGV != 0){
	print "perl find_cand4spair.pl stem[train, dev, test]\n";
	exit;
}
my $stem = shift @ARGV;
unless($stem =~ /(?:MNLI|MRPC|QNLI|QQP|RTE|STS-B).train/ or 
	   $stem =~ /(?:MNLI|MRPC|QNLI|QQP|RTE|STS-B).dev/){
	print "perl find_cand4spair.pl stem[train, dev, test]\n";
	exit;
}

my %sent = ();
my %np = ();
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
			if($cont =~ /SPACE/){
				$cont =~ s/  SPACE //g;
			}
			$cont =~ s/[ ]+/ /g;
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
	my $sent2 =  $sent{$key[$i]}{"2"};
	my @np1 = keys %{$np{$key[$i]}{"1"}};
	my @np2 = keys %{$np{$key[$i]}{"2"}};
	if($#np1 == -1 or $#np2 == -1){
		print FOUT "NO_NP1\t$key[$i]\t$sent1\t$sent2\t[MASK] $sent1\t[MASK] $sent2\n";
	}
	else{
		my ($lcs1, $lcs2, $mcs) = CheckLCS($key[$i]);
		my $idx1 = index($sent1, $lcs1);
		my $idx2 = index($sent2, $lcs2);
		if($lcs1 eq "null"){
			print FOUT "NO_NP2\t$key[$i]\t$sent1\t$sent2\t[MASK] $sent1\t[MASK] $sent2\n";
		}
		else{
			my $msent1 = GetMaskedStr($sent1, $lcs1);
			my $msent2 = GetMaskedStr($sent2, $lcs2);
			print FOUT "LCS\t$key[$i]\t$sent1\t$sent2\t$msent1\t$msent2\n"; 
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

sub CheckLCS{
	my ($id) = @_;
	my @np1 = sort keys %{$np{$id}{"1"}};
	my @np2 = sort keys %{$np{$id}{"2"}};
	my @tmp_lcs = ();
	for(my $i =0; $i <= $#np1; $i++){
		my $snp1 = $np{$id}{"1"}{$np1[$i]};
		my $max_lcs = "";
		my $np2_idx = -1;
		for(my $j =0; $j <= $#np2; $j++){
			my $snp2 = $np{$id}{"2"}{$np2[$j]};
			my $longest = LCSS::lcss($snp1, $snp2);
			$longest = "null" unless (defined $longest);
			if($longest ne "null"){
				$longest =~ s/[ ]+$//;
				$longest =~ s/^[ ]+//;
				my ($pass_flag, $mstr) = CheckValidity($longest, $snp1);
				my ($pass_flag2, $mstr2) = CheckValidity($longest, $snp2);
				if($pass_flag and $pass_flag2 and ($mstr eq $mstr2) and length($max_lcs) < length($mstr)){
					$max_lcs = $mstr;
					$np2_idx = $j;
				}
			}
		}
		if($np2_idx != -1){
			push @tmp_lcs, "$np1[$i]\t$np2[$np2_idx]\t$max_lcs";
		}
	}
	my $max_id = -1;
	my $max_ratio = -1;
	my $max_len   = -1;
	my $max_wlen   = -1;
	for(my $i =0; $i <= $#tmp_lcs; $i++){
		my @temp = split(/\t/,$tmp_lcs[$i]);
		my @temp2 = split(/ /,$temp[2]);
		my $lcs_len = length($temp[2]);
		my $lcs_wlen = $#temp2+1;
		my $ratio = (GetRatio($temp[0],$temp[2]) + GetRatio($temp[1], $temp[2]))/2;

		if($max_wlen < $#temp2+1){
			$max_wlen = $#temp2+1;
			$max_len = length($temp[2]);
			$max_ratio = $ratio;
			$max_id = $i;
		}
		elsif($max_wlen == $#temp2+1){
			if($max_len < length($temp[2])){
				$max_len = length($temp[2]);
				$max_ratio = $ratio;
				$max_id = $i;
			}
			elsif($max_len == length($temp[2])){
				if($max_ratio < $ratio){
					$max_ratio = $ratio;
					$max_id = $i;
				}
			}
		}
	}
	my $lcs1 = "null";
	my $lcs2 = "null";
	my $mcs  = "null";
	if($max_id != -1){
		my @temp = split(/\t/,$tmp_lcs[$max_id]);
		$lcs1 = $temp[0];
		$lcs2 = $temp[1];
		$mcs = $temp[2];
		my $elcs1 = GetExactMatch($lcs1,$mcs);
		my $elcs2 = GetExactMatch($lcs2,$mcs);
		$lcs1 = $elcs1;
		$lcs2 = $elcs2;
	}
	return ($lcs1, $lcs2, $mcs);
}

sub GetExactMatch{
	my ($ls, $cs) = @_;
	my @temp1 = split(/ /,$ls);
	my @temp2 = split(/ /,$cs);
	my @ltemp1 = split(/ /,$ls);
	my @tag = ();
	for(my $i =0; $i <= $#ltemp1; $i++){
		$ltemp1[$i] =~ tr/A-Z/a-z/;
		push @tag,0;
	}
	my $flag = 0;
	my $match = 0;
	my $cidx = 0;
	my $start = -1;
	my $end   = -1;
	for(my $i =0; $i <= $#temp2; $i++){
		my $src = $temp2[$i];
		
		$src =~ tr/A-Z/a-z/;
		for(my $j = $cidx; $j <= $#temp1; $j++){
			my $tgt = $ltemp1[$j];
			if($src eq $tgt){
				$tag[$j] = 1;
				$cidx = $j;
				$match++;
				$start = $cidx if $start == -1;
				$end   = $cidx;
				last;
			}
		}
	}
	$flag = 1 if $match == $#temp2+1;
	if($flag == 0){
		print "Can't find exact match for\n$ls\n$cs\n";
		for(my $i = 0; $i <= $#temp1; $i++){
			print "$temp1[$i]: $tag[$i]\n";
		}
		exit;
	}
	if($end-$start < 0){
		print "Can't find substring for $ls\t$cs\n";
		exit;
	}
	my $rstr = join(" ",@temp1[$start..$end]);
	return $rstr;
}

sub GetRatio{
	my ($str1, $str2) = @_;
	my @ostr = split(/ /,$str1);
	$str2 =~ s/[\\\/\*\[\]\$\+\(\)\{\}]//g;
	$str1 =~ s/$str2//g;
	$str1 =~ s/^[ ]+//g;
	$str1 =~ s/[ ]+$//g;
	my @mstr = split(/ /,$str1);
	my $ratio = ($#mstr+1)/($#ostr+1);
	return $ratio;
}

sub CheckValidity{
	my ($ls, $np_str) = @_;
	my $res_str = "null";
	my $idx = index($np_str, $ls);
	my @ls_token = split(/ /,$ls);
	# check left
	my $left_penalty = 0;
	my $flag = 0;
	if($idx != 0){ # first character
		my $left_c = substr($np_str, $idx-1, 1);
		if($left_c ne " "){
			$left_penalty = 1;
		}
	}
	# check right
	my $right_penalty = 0;
	my $len_ls = length($ls);
	my $len_np = length($np_str);
	my $schar = substr($np_str, $idx, 1);
	if($idx + length($ls) != length($np_str)){
		my $right_c = substr($np_str, $idx+length($ls), 1);
		if($right_c ne " "){
			$right_penalty = 1;
		}
	}
	my $dummy = shift @ls_token if $left_penalty;
	if($#ls_token != -1 and $right_penalty){
		$dummy = pop @ls_token;
	}
	if($#ls_token != -1){
		$res_str = join(" ",@ls_token);
		$flag = 1;
	}

	return ($flag, $res_str);
}

