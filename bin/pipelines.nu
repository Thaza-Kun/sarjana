export-env {	
	$env.OUTDATA = D:\home\kerja\sarjana\data\new\
	$env.EXPOSURE = D:\home\data\sourced\CHIMEFRB\exposure\daily\
}

export def "run exposure" [dir: path, exposures: path, limit: int = 3] {
	ls $dir 
		| where type == dir
		| get name 
		| filter {|a| 
			(ls $a 
				| get name 
				| path parse 
				| get stem 
				| 'exposure-UL' in $in
			) == false 
			} 
		| first $limit
		| each {|i|
			echo $'($i | path parse | get stem)';
			let ra = (open ($i | path join 'ra.txt')); echo $ra;
			let dec = (open ($i | path join 'dec.txt')); echo $dec;
			python get-exposure.py --dir $exposures --begin 2018-08-28 --end 2021-05-01 --out ($i | path join 'exposure-UL.csv') --ra $ra --dec $dec
			}
}
