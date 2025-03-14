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
			python get_exposure.py --dir $exposures --begin 2018-08-28 --end 2021-05-01 --out ($i | path join 'exposure-UL.csv') --ra $ra --dec $dec
			}
}

export def "test exposure" [name: string, dir: path, exposures: path] {
	echo $name;
	let ra = (open ($dir | path join $name 'ra.txt')); echo $ra;
	let dec = (open ($dir | path join $name 'dec.txt')); echo $dec;
	python get_exposure.py --dir $exposures --begin 2019-09-01 --end 2019-09-30 --out ($dir | path join $name 'exposure-UL-some.csv') --ra $ra --dec $dec
}

export def "pdgram eval" [name: string, folder: string, datadir: string, ngrid: float = 1.0, size: float = 1.0] {
	echo $name;
	python eval_composite_periodogram.py --name $name --simname $folder --datadir $datadir --size $size --snr 1 --ngrid $ngrid;
	python plot_composite_periodogram_single.py --name $name --simname $folder --size $size;
	python plot_periodogram_stack.py --name $name --simname $folder --size $size;
}

export def "pdgram test" [folder: string, datadir: string, size: float, name: string = "FRB20180916B"] {
	echo $name;
	python eval_composite_periodogram.py --name $name --simname $folder --datadir $datadir --size $size --snr 1;
	python plot_composite_periodogram_single.py --name $name --simname $folder --size $size;
	python plot_periodogram_stack.py --name $name --simname $folder --size $size;
}