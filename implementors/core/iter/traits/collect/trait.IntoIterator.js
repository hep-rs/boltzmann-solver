(function() {var implementors = {};
implementors["itertools"] = [{"text":"impl&lt;'a, K, I, F&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/iter/traits/collect/trait.IntoIterator.html\" title=\"trait core::iter::traits::collect::IntoIterator\">IntoIterator</a> for &amp;'a <a class=\"struct\" href=\"itertools/structs/struct.GroupBy.html\" title=\"struct itertools::structs::GroupBy\">GroupBy</a>&lt;K, I, F&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;I: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/iter/traits/iterator/trait.Iterator.html\" title=\"trait core::iter::traits::iterator::Iterator\">Iterator</a>,<br>&nbsp;&nbsp;&nbsp;&nbsp;I::<a class=\"type\" href=\"https://doc.rust-lang.org/nightly/core/iter/traits/iterator/trait.Iterator.html#associatedtype.Item\" title=\"type core::iter::traits::iterator::Iterator::Item\">Item</a>: 'a,<br>&nbsp;&nbsp;&nbsp;&nbsp;F: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/ops/function/trait.FnMut.html\" title=\"trait core::ops::function::FnMut\">FnMut</a>(&amp;I::<a class=\"type\" href=\"https://doc.rust-lang.org/nightly/core/iter/traits/iterator/trait.Iterator.html#associatedtype.Item\" title=\"type core::iter::traits::iterator::Iterator::Item\">Item</a>) -&gt; K,<br>&nbsp;&nbsp;&nbsp;&nbsp;K: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/cmp/trait.PartialEq.html\" title=\"trait core::cmp::PartialEq\">PartialEq</a>,&nbsp;</span>","synthetic":false,"types":["itertools::groupbylazy::GroupBy"]},{"text":"impl&lt;'a, I&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/iter/traits/collect/trait.IntoIterator.html\" title=\"trait core::iter::traits::collect::IntoIterator\">IntoIterator</a> for &amp;'a <a class=\"struct\" href=\"itertools/structs/struct.IntoChunks.html\" title=\"struct itertools::structs::IntoChunks\">IntoChunks</a>&lt;I&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;I: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/iter/traits/iterator/trait.Iterator.html\" title=\"trait core::iter::traits::iterator::Iterator\">Iterator</a>,<br>&nbsp;&nbsp;&nbsp;&nbsp;I::<a class=\"type\" href=\"https://doc.rust-lang.org/nightly/core/iter/traits/iterator/trait.Iterator.html#associatedtype.Item\" title=\"type core::iter::traits::iterator::Iterator::Item\">Item</a>: 'a,&nbsp;</span>","synthetic":false,"types":["itertools::groupbylazy::IntoChunks"]},{"text":"impl&lt;'a, I&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/iter/traits/collect/trait.IntoIterator.html\" title=\"trait core::iter::traits::collect::IntoIterator\">IntoIterator</a> for &amp;'a <a class=\"struct\" href=\"itertools/structs/struct.RcIter.html\" title=\"struct itertools::structs::RcIter\">RcIter</a>&lt;I&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;I: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/iter/traits/iterator/trait.Iterator.html\" title=\"trait core::iter::traits::iterator::Iterator\">Iterator</a>,&nbsp;</span>","synthetic":false,"types":["itertools::rciter_impl::RcIter"]}];
implementors["ndarray"] = [{"text":"impl&lt;'a, S, D&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/iter/traits/collect/trait.IntoIterator.html\" title=\"trait core::iter::traits::collect::IntoIterator\">IntoIterator</a> for &amp;'a <a class=\"struct\" href=\"ndarray/struct.ArrayBase.html\" title=\"struct ndarray::ArrayBase\">ArrayBase</a>&lt;S, D&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;D: <a class=\"trait\" href=\"ndarray/trait.Dimension.html\" title=\"trait ndarray::Dimension\">Dimension</a>,<br>&nbsp;&nbsp;&nbsp;&nbsp;S: <a class=\"trait\" href=\"ndarray/trait.Data.html\" title=\"trait ndarray::Data\">Data</a>,&nbsp;</span>","synthetic":false,"types":["ndarray::ArrayBase"]},{"text":"impl&lt;'a, S, D&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/iter/traits/collect/trait.IntoIterator.html\" title=\"trait core::iter::traits::collect::IntoIterator\">IntoIterator</a> for &amp;'a mut <a class=\"struct\" href=\"ndarray/struct.ArrayBase.html\" title=\"struct ndarray::ArrayBase\">ArrayBase</a>&lt;S, D&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;D: <a class=\"trait\" href=\"ndarray/trait.Dimension.html\" title=\"trait ndarray::Dimension\">Dimension</a>,<br>&nbsp;&nbsp;&nbsp;&nbsp;S: <a class=\"trait\" href=\"ndarray/trait.DataMut.html\" title=\"trait ndarray::DataMut\">DataMut</a>,&nbsp;</span>","synthetic":false,"types":["ndarray::ArrayBase"]},{"text":"impl&lt;'a, A, D&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/iter/traits/collect/trait.IntoIterator.html\" title=\"trait core::iter::traits::collect::IntoIterator\">IntoIterator</a> for <a class=\"type\" href=\"ndarray/type.ArrayView.html\" title=\"type ndarray::ArrayView\">ArrayView</a>&lt;'a, A, D&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;D: <a class=\"trait\" href=\"ndarray/trait.Dimension.html\" title=\"trait ndarray::Dimension\">Dimension</a>,&nbsp;</span>","synthetic":false,"types":["ndarray::ArrayView"]},{"text":"impl&lt;'a, A, D&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/iter/traits/collect/trait.IntoIterator.html\" title=\"trait core::iter::traits::collect::IntoIterator\">IntoIterator</a> for <a class=\"type\" href=\"ndarray/type.ArrayViewMut.html\" title=\"type ndarray::ArrayViewMut\">ArrayViewMut</a>&lt;'a, A, D&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;D: <a class=\"trait\" href=\"ndarray/trait.Dimension.html\" title=\"trait ndarray::Dimension\">Dimension</a>,&nbsp;</span>","synthetic":false,"types":["ndarray::ArrayViewMut"]},{"text":"impl&lt;D&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/iter/traits/collect/trait.IntoIterator.html\" title=\"trait core::iter::traits::collect::IntoIterator\">IntoIterator</a> for <a class=\"struct\" href=\"ndarray/iter/struct.Indices.html\" title=\"struct ndarray::iter::Indices\">Indices</a>&lt;D&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;D: <a class=\"trait\" href=\"ndarray/trait.Dimension.html\" title=\"trait ndarray::Dimension\">Dimension</a>,&nbsp;</span>","synthetic":false,"types":["ndarray::indexes::Indices"]},{"text":"impl&lt;'a, A, D&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/iter/traits/collect/trait.IntoIterator.html\" title=\"trait core::iter::traits::collect::IntoIterator\">IntoIterator</a> for <a class=\"struct\" href=\"ndarray/iter/struct.ExactChunks.html\" title=\"struct ndarray::iter::ExactChunks\">ExactChunks</a>&lt;'a, A, D&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;D: <a class=\"trait\" href=\"ndarray/trait.Dimension.html\" title=\"trait ndarray::Dimension\">Dimension</a>,<br>&nbsp;&nbsp;&nbsp;&nbsp;A: 'a,&nbsp;</span>","synthetic":false,"types":["ndarray::iterators::chunks::ExactChunks"]},{"text":"impl&lt;'a, A, D&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/iter/traits/collect/trait.IntoIterator.html\" title=\"trait core::iter::traits::collect::IntoIterator\">IntoIterator</a> for <a class=\"struct\" href=\"ndarray/iter/struct.ExactChunksMut.html\" title=\"struct ndarray::iter::ExactChunksMut\">ExactChunksMut</a>&lt;'a, A, D&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;D: <a class=\"trait\" href=\"ndarray/trait.Dimension.html\" title=\"trait ndarray::Dimension\">Dimension</a>,<br>&nbsp;&nbsp;&nbsp;&nbsp;A: 'a,&nbsp;</span>","synthetic":false,"types":["ndarray::iterators::chunks::ExactChunksMut"]},{"text":"impl&lt;'a, A, D&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/iter/traits/collect/trait.IntoIterator.html\" title=\"trait core::iter::traits::collect::IntoIterator\">IntoIterator</a> for <a class=\"struct\" href=\"ndarray/iter/struct.Lanes.html\" title=\"struct ndarray::iter::Lanes\">Lanes</a>&lt;'a, A, D&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;D: <a class=\"trait\" href=\"ndarray/trait.Dimension.html\" title=\"trait ndarray::Dimension\">Dimension</a>,&nbsp;</span>","synthetic":false,"types":["ndarray::iterators::lanes::Lanes"]},{"text":"impl&lt;'a, A, D&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/iter/traits/collect/trait.IntoIterator.html\" title=\"trait core::iter::traits::collect::IntoIterator\">IntoIterator</a> for <a class=\"struct\" href=\"ndarray/iter/struct.LanesMut.html\" title=\"struct ndarray::iter::LanesMut\">LanesMut</a>&lt;'a, A, D&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;D: <a class=\"trait\" href=\"ndarray/trait.Dimension.html\" title=\"trait ndarray::Dimension\">Dimension</a>,&nbsp;</span>","synthetic":false,"types":["ndarray::iterators::lanes::LanesMut"]},{"text":"impl&lt;'a, A, D&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/iter/traits/collect/trait.IntoIterator.html\" title=\"trait core::iter::traits::collect::IntoIterator\">IntoIterator</a> for <a class=\"struct\" href=\"ndarray/iter/struct.Windows.html\" title=\"struct ndarray::iter::Windows\">Windows</a>&lt;'a, A, D&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;D: <a class=\"trait\" href=\"ndarray/trait.Dimension.html\" title=\"trait ndarray::Dimension\">Dimension</a>,<br>&nbsp;&nbsp;&nbsp;&nbsp;A: 'a,&nbsp;</span>","synthetic":false,"types":["ndarray::iterators::windows::Windows"]},{"text":"impl&lt;'a&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/iter/traits/collect/trait.IntoIterator.html\" title=\"trait core::iter::traits::collect::IntoIterator\">IntoIterator</a> for &amp;'a <a class=\"struct\" href=\"ndarray/struct.IxDynImpl.html\" title=\"struct ndarray::IxDynImpl\">IxDynImpl</a>","synthetic":false,"types":["ndarray::dimension::dynindeximpl::IxDynImpl"]}];

            if (window.register_implementors) {
                window.register_implementors(implementors);
            } else {
                window.pending_implementors = implementors;
            }
        })()