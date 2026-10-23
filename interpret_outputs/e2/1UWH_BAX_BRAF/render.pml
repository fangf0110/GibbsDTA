reinitialize
load pocket_attention.pdb, cplx
hide everything
bg_color white
set ray_opaque_background, 0
set cartoon_transparency, 0.15
show cartoon, cplx and polymer
spectrum b, white_red, cplx and polymer, minimum=0, maximum=100
show sticks, cplx and resn BAX
util.cbac cplx and resn BAX
set stick_radius, 0.22, cplx and resn BAX
select topres, cplx and polymer and chain A and resi 528+530+533+532+513+582+511+515
show sticks, topres and not name N+C+O
color tv_red, topres and elem C
set stick_radius, 0.16, topres
set label_size, 18
set label_color, black
label topres and name CA, "%s%s" % (resn, resi)
orient cplx and resn BAX
zoom cplx and resn BAX, 9
set ray_trace_mode, 1
png pocket_attention.png, width=2400, height=1800, dpi=300, ray=1
save pocket_attention.pse
