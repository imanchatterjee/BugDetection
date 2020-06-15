/*global console document asmjs_perf_in_place genflatcode_asmjs_dftreal1024flat asmjs_remove_in_place_from_gen asmjs_perf_compare_gen_dft asmjs_perf_format_one_result dftreal1024flat_asmjsGen*/

function asmjs_perf_in_place()
{
    // Make sure we have an implementation
    genflatcode_asmjs_dftreal1024flat();

    // For a meaningful comparison we need to deactivate asm.js
    // because asm.js would compile the Typed Array version,
    // but not the normal array version.
    var with_new_output_array = {
        fun : dftreal1024flat.getDirect()  // Generated by `flatorize`
        , normal_array : true 
    }
    
    , with_in_place = {
        gen : asmjs_remove_typed_array_from_gen( asmjs_remove_use_asm_from_gen( dftreal1024flat_asmjsGen ) )
        , normal_array : true
    }
       
    , result = asmjs_perf_compare_gen_dft( 
        1024
        , { with_new_output_array : with_new_output_array
            ,  with_in_place      : with_in_place
          }
    )
    ;

    console.log( 'asmjs_perf_in_place: result:', result );

    var outnode = asmjs_perf_in_place.outnode;
    if (!outnode)
    {
        outnode = asmjs_perf_in_place.outnode = document.getElementById( 'asmjs_perf_in_place_output' );
        outnode.innerHTML = '';
    }
    
    outnode.innerHTML += asmjs_perf_format_one_result( 'with new output array', result.with_new_output_array ) + '\n'
        + asmjs_perf_format_one_result( 'with   in-place array', result.with_in_place ) + '\n'
        + '-> speedup: ' 
        + asmjs_perf_prop_2_percent_string( result.with_in_place.speed_iter_per_sec / result.with_new_output_array.speed_iter_per_sec ) + '\n'
        + '\n'
    ;
}