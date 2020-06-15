/**
 * @author thatcher
 */
(function($){    
	//-------------------------------------------------------------------------------------//
	//  -   ENVIRONMENTAL CONFIGURATION   -
	//______________________________________________________________________________________//
	$.config("env", {
	    defaults:{
			templates:'app/views/templates',
	        rest: {
	            SERVICE: "Any",
	            URL: "/rest",
	            AJAX: "jQuery"
	        }
	    },
	    //-------------------------------------------------------------------------------------//
	    //  -   DEVELOPMENT CONFIGURATION   -
	    //______________________________________________________________________________________//
	    dev:{
	        client:{
	        }
	    },
	    //-------------------------------------------------------------------------------------//
	    //  -   PRODUCTION CONFIGURATION   -
	    //______________________________________________________________________________________//
	    prod:{
	        client:{
	        }
	    },
	    //-------------------------------------------------------------------------------------//
	    //  -   TEST CONFIGURATION   -
	    //______________________________________________________________________________________//
	    test:{
	        client:{
	        }
	    }
	}); 
    
})(jQuery);
    