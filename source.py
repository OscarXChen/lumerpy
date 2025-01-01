from .fdtd_manager import get_fdtd_instance

u = 1e-6


def add_source_dipole(x=0, y=0, z=0, wavelength_start=1.55 * u, wavelength_stop=1.55 * u, angle_phase=0, angle_theta=0,
					  angle_phi=0, dipole_type="Electric dipole"):
	FD = get_fdtd_instance()
	ob_point = FD.adddipole()
	FD.set("x", x)
	FD.set("y", y)
	FD.set("z", z)
	FD.set("wavelength start", wavelength_start)
	FD.set("wavelength stop", wavelength_stop)
	FD.set("phi", angle_phi)
	FD.set("phase", angle_phase)
	FD.set("theta", angle_theta)
	FD.set("dipole type", dipole_type)
	return ob_point


def add_source_gaussian(x_min=0, x_max=0, y_min=0, y_max=0, z_min=0, z_max=0, injection_axis="x", direction="forward",
						polarization_angle=0, angle_theta=0, angle_phi=0, wavelength_start=1.55 * u,
						wavelength_stop=1.55 * u, source_shape="Gaussian"):
	FD = get_fdtd_instance()
	ob_source_gasussian = FD.addgaussian()
	FD.set("injection axis", injection_axis)
	if injection_axis == "x":
		if x_min != x_max and x_max != 0:
			print("警告！请检查高斯/平面波光源x位置是否设置正确")
		FD.set("x", x_min)
		FD.set("y min", y_min)
		FD.set("y max", y_max)
		FD.set("z min", z_min)
		FD.set("z max", z_max)
	if injection_axis == "y":
		if y_min != y_max and y_max != 0:
			print("警告！请检查高斯/平面波光源y位置是否设置正确")
		FD.set("y", y_min)
		FD.set("x min", x_min)
		FD.set("x max", x_max)
		FD.set("z min", z_min)
		FD.set("z max", z_max)
	if injection_axis == "z":
		if z_min != z_max and z_max != 0:
			print("警告！请检查高斯/平面波光源z位置是否设置正确")
		FD.set("z", z_min)
		FD.set("x min", x_min)
		FD.set("x max", x_max)
		FD.set("y min", y_min)
		FD.set("y max", y_max)

	FD.set("direction", direction)
	FD.set("wavelength start", wavelength_start)
	FD.set("wavelength stop", wavelength_stop)
	FD.set("angle phi", angle_phi)
	FD.set("angle theta", angle_theta)
	FD.set("polarization angle", polarization_angle)
	FD.set("source shape", source_shape)
	return ob_source_gasussian


def add_source_plane(x_min=0, x_max=0, y_min=0, y_max=0, z_min=0, z_max=0, injection_axis="x", direction="forward",
					 polarization_angle=0, angle_theta=0, angle_phi=0, wavelength_start=1.55 * u,
					 wavelength_stop=1.55 * u, source_shape="Plane wave"):
	ob_plane = add_source_gaussian(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, z_min=z_min, z_max=z_max,
								   injection_axis=injection_axis, direction=direction,
								   polarization_angle=polarization_angle, angle_theta=angle_theta, angle_phi=angle_phi,
								   wavelength_start=wavelength_start, wavelength_stop=wavelength_stop,
								   source_shape=source_shape)
	return ob_plane


def add_source_mode(x_min=0, x_max=0, y_min=0, y_max=0, z_min=0, z_max=0, wavelength_start=1.55 * u,
					wavelength_stop=1.55 * u, injection_axis="x", direction="forward"):
	FD = get_fdtd_instance()
	ob_source_mode_plane = FD.addmode()
	FD.set("injection axis", injection_axis)
	if injection_axis == "x":
		if x_min != x_max and x_max != 0:
			print("警告！请检查点光源x位置是否设置正确")
		FD.set("x", x_min)
		FD.set("y min", y_min)
		FD.set("y max", y_max)
		FD.set("z min", z_min)
		FD.set("z max", z_max)
	if injection_axis == "y":
		if y_min != y_max and y_max != 0:
			print("警告！请检查点光源y位置是否设置正确")
		FD.set("y", y_min)
		FD.set("x min", x_min)
		FD.set("x max", x_max)
		FD.set("z min", z_min)
		FD.set("z max", z_max)
	if injection_axis == "z":
		if z_min != z_max and z_max != 0:
			print("警告！请检查点光源y位置是否设置正确")
		FD.set("z", z_min)
		FD.set("x min", x_min)
		FD.set("x max", x_max)
		FD.set("y min", y_min)
		FD.set("y max", y_max)

	FD.set("direction", direction)
	FD.set("wavelength start", wavelength_start)
	FD.set("wavelength stop", wavelength_stop)
	# FD.set("angle phi", 90)
	return ob_source_mode_plane
