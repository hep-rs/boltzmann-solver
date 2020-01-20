var N = null;var sourcesIndex = {};
sourcesIndex["boltzmann_solver"] = {"name":"","dirs":[{"name":"model","files":["data.rs","interaction.rs","particle.rs","standard_model.rs"]},{"name":"solver","dirs":[{"name":"tableau","files":["rk21.rs","rk32.rs","rk43.rs","rk54.rs","rk65.rs","rk76.rs","rk87.rs","rk98.rs"]}],"files":["context.rs","options.rs","solver.rs","tableau.rs"]},{"name":"utilities","files":["spline.rs"]}],"files":["constants.rs","lib.rs","model.rs","prelude.rs","solver.rs","statistic.rs","utilities.rs"]};
sourcesIndex["cfg_if"] = {"name":"","files":["lib.rs"]};
sourcesIndex["crossbeam_deque"] = {"name":"","files":["lib.rs"]};
sourcesIndex["crossbeam_epoch"] = {"name":"","dirs":[{"name":"sync","files":["list.rs","mod.rs","queue.rs"]}],"files":["atomic.rs","collector.rs","default.rs","deferred.rs","epoch.rs","guard.rs","internal.rs","lib.rs"]};
sourcesIndex["crossbeam_queue"] = {"name":"","files":["array_queue.rs","err.rs","lib.rs","seg_queue.rs"]};
sourcesIndex["crossbeam_utils"] = {"name":"","dirs":[{"name":"atomic","files":["atomic_cell.rs","consume.rs","mod.rs","seq_lock.rs"]},{"name":"sync","files":["mod.rs","parker.rs","sharded_lock.rs","wait_group.rs"]}],"files":["backoff.rs","cache_padded.rs","lib.rs","thread.rs"]};
sourcesIndex["either"] = {"name":"","files":["lib.rs"]};
sourcesIndex["itertools"] = {"name":"","dirs":[{"name":"adaptors","files":["mod.rs","multi_product.rs"]}],"files":["combinations.rs","combinations_with_replacement.rs","concat_impl.rs","cons_tuples_impl.rs","diff.rs","either_or_both.rs","exactly_one_err.rs","format.rs","free.rs","group_map.rs","groupbylazy.rs","impl_macros.rs","intersperse.rs","kmerge_impl.rs","lazy_buffer.rs","lib.rs","merge_join.rs","minmax.rs","multipeek_impl.rs","pad_tail.rs","peeking_take_while.rs","permutations.rs","process_results_impl.rs","put_back_n_impl.rs","rciter_impl.rs","repeatn.rs","size_hint.rs","sources.rs","tee.rs","tuple_impl.rs","unique_impl.rs","with_position.rs","zip_eq_impl.rs","zip_longest.rs","ziptuple.rs"]};
sourcesIndex["lazy_static"] = {"name":"","files":["inline_lazy.rs","lib.rs"]};
sourcesIndex["libc"] = {"name":"","dirs":[{"name":"unix","dirs":[{"name":"linux_like","dirs":[{"name":"linux","dirs":[{"name":"gnu","dirs":[{"name":"b64","dirs":[{"name":"x86_64","files":["align.rs","mod.rs","not_x32.rs"]}],"files":["mod.rs"]}],"files":["align.rs","mod.rs"]}],"files":["align.rs","mod.rs"]}],"files":["mod.rs"]}],"files":["align.rs","mod.rs"]}],"files":["fixed_width_ints.rs","lib.rs","macros.rs"]};
sourcesIndex["log"] = {"name":"","files":["lib.rs","macros.rs"]};
sourcesIndex["matrixmultiply"] = {"name":"","dirs":[{"name":"x86","files":["macros.rs","mod.rs"]}],"files":["aligned_alloc.rs","archparam.rs","debugmacros.rs","dgemm_kernel.rs","gemm.rs","kernel.rs","lib.rs","loopmacros.rs","sgemm_kernel.rs","util.rs"]};
sourcesIndex["memoffset"] = {"name":"","files":["lib.rs","offset_of.rs","span_of.rs"]};
sourcesIndex["ndarray"] = {"name":"","dirs":[{"name":"dimension","files":["axes.rs","axis.rs","conversion.rs","dim.rs","dimension_trait.rs","dynindeximpl.rs","macros.rs","mod.rs","ndindex.rs","remove_axis.rs"]},{"name":"extension","files":["nonnull.rs"]},{"name":"impl_views","files":["constructors.rs","conversions.rs","indexing.rs","mod.rs","splitting.rs"]},{"name":"iterators","files":["chunks.rs","iter.rs","lanes.rs","macros.rs","mod.rs","windows.rs"]},{"name":"layout","files":["layoutfmt.rs","mod.rs"]},{"name":"linalg","files":["impl_linalg.rs","mod.rs"]},{"name":"numeric","files":["impl_numeric.rs","mod.rs"]},{"name":"parallel","files":["impl_par_methods.rs","into_impls.rs","mod.rs","par.rs","zipmacro.rs"]},{"name":"zip","files":["mod.rs","zipmacro.rs"]}],"files":["aliases.rs","arrayformat.rs","arraytraits.rs","data_traits.rs","error.rs","extension.rs","free_functions.rs","geomspace.rs","impl_1d.rs","impl_2d.rs","impl_clone.rs","impl_constructors.rs","impl_cow.rs","impl_dyn.rs","impl_methods.rs","impl_ops.rs","impl_owned_array.rs","impl_raw_views.rs","indexes.rs","lib.rs","linalg_traits.rs","linspace.rs","logspace.rs","macro_utils.rs","numeric_util.rs","prelude.rs","private.rs","shape_builder.rs","slice.rs","stacking.rs"]};
sourcesIndex["num_complex"] = {"name":"","files":["cast.rs","lib.rs","pow.rs"]};
sourcesIndex["num_cpus"] = {"name":"","files":["lib.rs"]};
sourcesIndex["num_integer"] = {"name":"","files":["lib.rs","roots.rs"]};
sourcesIndex["num_traits"] = {"name":"","dirs":[{"name":"ops","files":["checked.rs","inv.rs","mod.rs","mul_add.rs","saturating.rs","wrapping.rs"]}],"files":["bounds.rs","cast.rs","float.rs","identities.rs","int.rs","lib.rs","macros.rs","pow.rs","real.rs","sign.rs"]};
sourcesIndex["quadrature"] = {"name":"","dirs":[{"name":"clenshaw_curtis","files":["constants.rs","mod.rs"]},{"name":"double_exponential","files":["constants.rs","mod.rs"]}],"files":["lib.rs"]};
sourcesIndex["rawpointer"] = {"name":"","files":["lib.rs"]};
sourcesIndex["rayon"] = {"name":"","dirs":[{"name":"collections","files":["binary_heap.rs","btree_map.rs","btree_set.rs","hash_map.rs","hash_set.rs","linked_list.rs","mod.rs","vec_deque.rs"]},{"name":"compile_fail","files":["cannot_collect_filtermap_data.rs","cannot_zip_filtered_data.rs","cell_par_iter.rs","mod.rs","must_use.rs","no_send_par_iter.rs","rc_par_iter.rs"]},{"name":"iter","dirs":[{"name":"collect","files":["consumer.rs","mod.rs"]},{"name":"find_first_last","files":["mod.rs"]},{"name":"plumbing","files":["mod.rs"]}],"files":["chain.rs","chunks.rs","cloned.rs","copied.rs","empty.rs","enumerate.rs","extend.rs","filter.rs","filter_map.rs","find.rs","flat_map.rs","flatten.rs","fold.rs","for_each.rs","from_par_iter.rs","inspect.rs","interleave.rs","interleave_shortest.rs","intersperse.rs","len.rs","map.rs","map_with.rs","mod.rs","multizip.rs","noop.rs","once.rs","panic_fuse.rs","par_bridge.rs","product.rs","reduce.rs","repeat.rs","rev.rs","skip.rs","splitter.rs","sum.rs","take.rs","try_fold.rs","try_reduce.rs","try_reduce_with.rs","unzip.rs","update.rs","while_some.rs","zip.rs","zip_eq.rs"]},{"name":"slice","files":["mergesort.rs","mod.rs","quicksort.rs"]}],"files":["delegate.rs","lib.rs","math.rs","option.rs","par_either.rs","prelude.rs","private.rs","range.rs","range_inclusive.rs","result.rs","split_producer.rs","str.rs","vec.rs"]};
sourcesIndex["rayon_core"] = {"name":"","dirs":[{"name":"compile_fail","files":["mod.rs","quicksort_race1.rs","quicksort_race2.rs","quicksort_race3.rs","rc_return.rs","rc_upvar.rs","scope_join_bad.rs"]},{"name":"join","files":["mod.rs"]},{"name":"scope","files":["mod.rs"]},{"name":"sleep","files":["mod.rs"]},{"name":"spawn","files":["mod.rs"]},{"name":"thread_pool","files":["mod.rs"]}],"files":["job.rs","latch.rs","lib.rs","log.rs","private.rs","registry.rs","unwind.rs","util.rs"]};
sourcesIndex["scopeguard"] = {"name":"","files":["lib.rs"]};
sourcesIndex["special_functions"] = {"name":"","dirs":[{"name":"approximations","files":["linear.rs"]},{"name":"bessel","files":["i0.rs","i1.rs","i2.rs","i3.rs","i4.rs","i5.rs","i6.rs","i7.rs","i8.rs","i9.rs","j0.rs","j1.rs","j2.rs","j3.rs","j4.rs","j5.rs","j6.rs","j7.rs","j8.rs","j9.rs","k0.rs","k1.rs","k1_on_k2.rs","k2.rs","k3.rs","k4.rs","k5.rs","k6.rs","k7.rs","k8.rs","k9.rs","y0.rs","y1.rs","y2.rs","y3.rs","y4.rs","y5.rs","y6.rs","y7.rs","y8.rs","y9.rs"]},{"name":"other","dirs":[{"name":"polylog","files":["li2.rs","li3.rs","li4.rs","li5.rs","li6.rs","li7.rs","li8.rs","li9.rs"]}],"files":["harmonic_number.rs","polylog.rs"]},{"name":"particle_statistics","files":["bose_einstein_massive.rs","bose_einstein_massless.rs","bose_einstein_normalized.rs","fermi_dirac_massive.rs","fermi_dirac_massless.rs","fermi_dirac_normalized.rs","pave.rs"]},{"name":"utilities","files":["mod.rs"]}],"files":["approximations.rs","basic.rs","bessel.rs","lib.rs","other.rs","particle_statistics.rs"]};
createSourceSidebar();