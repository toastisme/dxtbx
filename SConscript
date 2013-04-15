import os
import libtbx.load_env
Import("env_etc")

env_etc.dxtbx_dist = libtbx.env.dist_path("dxtbx")
env_etc.dxtbx_include = os.path.dirname(env_etc.dxtbx_dist)
if (not env_etc.no_boost_python and hasattr(env_etc, "boost_adaptbx_include")):
  Import("env_no_includes_boost_python_ext")
  env = env_no_includes_boost_python_ext.Clone()
  env_etc.enable_more_warnings(env=env)
  env_etc.include_registry.append(
    env=env,
    paths=[
      env_etc.libtbx_include,
      env_etc.boost_adaptbx_include,
      env_etc.boost_include,
      env_etc.python_include,
      env_etc.dxtbx_include])
  env.Append(
	LIBS=env_etc.libm + [ 
	"scitbx_boost_python"])

  env.SharedLibrary(
    target="#lib/dxtbx_ext",
    source=[
      "boost_python/to_ewald_sphere_helpers.cc",
      "boost_python/ext.cpp"])
    
  env.SConscript('model/SConscript', exports={ 'env' : env })
