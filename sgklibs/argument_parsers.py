import argparse

from .low_level_utils import isint, isfloat

class CustomHelpFormatter(argparse.HelpFormatter):
    '''
    A slightly modified version of the HelpFormatter
    class from argparse that includes that defaults
    for options that have defaults.
    '''

    def _get_help_string(self, action):
        help = action.help
        if '%(default)' not in action.help:
            if action.default is not argparse.SUPPRESS:
                defaulting_nargs = [argparse.OPTIONAL, argparse.ZERO_OR_MORE]
                if action.option_strings or action.nargs in defaulting_nargs:
                    help += ' (default: %(default)s)'
        if '%(type)' not in action.help:
            if action.default is not argparse.SUPPRESS:
                defaulting_nargs = [argparse.OPTIONAL, argparse.ZERO_OR_MORE]
                if action.option_strings or action.nargs in defaulting_nargs:
                    help += ' (type: %(type)s)'
        return help

class FunctionParser:
    '''
    A class that takes in a function and makes it 
    callable via the command line.  That is, if you
    have a function:

    def my_function(required_variable, optional_variable1=a):
        ...

    then you can add:

    if __name__ == "__main__":
        kwargs = FunctionParser(my_function).parse_args()
        my_function(**kwargs)

    then call the function via the command line with 

    $ python file_name.py required_variable_value --optional_variable1=b

    obviously not suitable for all functions, but makes it easier to write
    main style functions.  also tries to intelligently parse different data
    types, includes things like lists, arrays, and dictionaries.

    this class will even go through the docstring of my_function (if included) 
    and pull out the description of each parameter (and the type of that parameter)
    to use for the help text of the parser and to parse into the proper datatype.
    if there are no docstrings, then default value types are used for the optinoal 
    arguments and required arguments will be passed following the rules in 
    parse_string
    '''

    def parse_string(self, val):
        '''
        given a string val, tries to parse that string into the
        most logical of datatypes.  

            * if it's a string that contains true or false, then you 
                will get the appropriate boolean

            * if it's a string that contains an integer, you'll get
                an integer

            * if it's a string that contains a float, you'll get the
                float

            * otherwise, you'll get the string with whitespace stripped off

        returns the processed value, as described above
        '''

        if val in ['True', 'true', True, 'TRUE']:
            return True
        elif val in ['False', 'false', False, 'FALSE']:
            return False

        #harder to pass an int check than a float check, so do that first
        elif isint(val):
            return int(val)
        elif isfloat(val):
            return float(val)
        else:
            return val.strip()

    def parse_list(self, string):
        '''
        wrapper around parse_string that takes in a 
        comma separated list of items and parses each
        in turn into a list.  optionally strips off [].
        '''

        lst = string.strip('[]').split(',')
        for ii, val in enumerate(lst):
            lst[ii] = self.parse_string(val)
        return lst

    def parse_array(self, string):
        '''
        wrapper around parse_list that interprets the output
        as a numpy array.  
        '''
        from numpy import array
        lst = self.parse_list(string)
        return array(lst)

    def parse_dict_general(self, string, limits=False, sep=';'):
        '''
        parses a string into a dictionary.  sep determines the 
        separater between key-val pairs; keys and values can 
        themselves be separated by =, :, or a space (will be 
        checked in the order).  can also handles dictionaries 
        that contain lists as items, though must be careful that 
        there are no =, :, or spaces in thelist itself (depending 
        on the adopted separator)

        for example:

            string = 'key1:val1; key2=val2; key3 [val31, val32, val33]'
            self.parse_dict_general(string, sep=':')
            >>> {'key1':'val1', 'key2':'val2', 'key3':['val31', 'val32', 'val33']}


        this function has optional support for a "limit dictionary",
        where the keys are properties and the values are length two
        lists that bound an allowed region.  see parse_limlist for 
        guidance on those.
        '''
        dictionary = {}
        items = string.strip('{}').split(sep)
        for it in items:
            it.strip()
            if not len(it):
                continue
            if '=' in it:
                key,val = it.split('=')
            elif ':' in it:
                key,val = it.split(':')
            elif ' ' in it:
                key,val = it.split()
            else:
                raise ValueError("no valid separator in {}".format(it))

            #parse as a list in case it is, then check afterwards if it wasn't:
            if limits:
                val = self.parse_limlist(val)
            else:
                val = self.parse_list(val)
                if len(val) == 1:   val = val[0]
            dictionary[key.strip()] = val
        return dictionary

    def parse_dict(self, string):
        '''
        parse a string into a dictionary that is *not* a limit dictionary
        '''
        return self.parse_dict_general(string, limits=False)

    def parse_limdict(self, string):
        '''
        parse a string into a dictionary is is a limit diciontary
        '''
        return self.parse_dict_general(string, limits=True)

    def parse_limlist(self, lims):
        '''
        parse a string into min/max limits with the following rules:
            * if a , is not included, float(lims) -- infinity is allowed
            * if the string begins with either ","" or "[,", the -np.inf  -- float(lims) is allowed
            * otherwise, splits on a comma and takes float of left and right
                of that comma as the allowed region (defaulting to inf
                for max allowed if there is one entry after splitting) 
        '''

        if ',' not in lims:
            lims = [float(lims), np.inf]
        elif lims[0] == ',' or lims[:2] =='[,':
            lims = [-np.inf, float(lims.strip('[,'))]
        else:
            lims = lims.strip('[] ').split(',')
            lims = [float(l) for l in lims if len(l)]
            if len(lims) == 1:
                lims.append(np.inf)
        return lims

    def mybool(self,val):
        #handle boolean conversions w/o telling argparse that's what I'm doing
        return self.parse_string(val)

    def __init__(self, function=None):
        import inspect, argparse
        from numpy import ndarray

        # valid keywords in the docstrings for different datatypes
        types = ['float', 'bool', 'boolean', 'int', 'integer', 'string',
                'str','list-like','list','array-like','array', 'dict', 
                'dictionary', 'dict-like', 'limit-dict', 'limdict','limitdict']

        # how we handle each datatype
        tdict = {'float':float, float:float,
            'bool':self.mybool, 'boolean':self.mybool, bool:self.mybool,
            'int':int, 'integer':int, int:int,
            'string':str, 'str':str, '':str, str:str, None:str, type(None):str,
            list: self.parse_list, 'list-like': self.parse_list, 'list': self.parse_list, 
            ndarray: self.parse_array, 'array-like': self.parse_array, 'array': self.parse_array, 
            dict: self.parse_dict, 'dict': self.parse_dict, 'dictionary': self.parse_dict, 'dict-like': self.parse_dict,
            'limit-dict': self.parse_limdict, 'limdict': self.parse_limdict, 'limitdict': self.parse_limdict}

        if function is None:
            # self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            self.parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)

        else:
            self.argspec = inspect.getfullargspec(function)
            self.args = list(self.argspec.args)
            if type(self.argspec.defaults) is type(None):
                self.defaults = []
            else:
                self.defaults = list(self.argspec.defaults)

            self.callstring = inspect.formatargspec(*self.argspec).strip('()')
            
            self.fullhelp = inspect.getdoc(function)
            if type(self.fullhelp) is type(None):
                self.fullhelp = ''

            self.parhelp = {}
            self.partypes = {}

            if 'param' in self.fullhelp:
                self.docstring = self.fullhelp.split('param')[0].strip(' \n')
            else:
                self.docstring = self.fullhelp

            with_defaults = len(self.defaults)
            nargs = len(self.args)
            no_defaults = nargs - with_defaults
            for ii in range(no_defaults):
                self.defaults.insert(0, 'nodef')

            self.helpdict = {}
            self.typedict = {}
            for ii, arg in enumerate(self.args):
                found = False
                if ':param '+arg+':' in self.fullhelp:
                    idx = self.fullhelp.index(':param '+arg+':')
                    parhelp = self.fullhelp[idx+len(':param '+arg+':'):].split(':param')[0]
                    
                    for t in types:
                        if ' '+t in parhelp or ':'+t in parhelp:
                            parhelp = parhelp.replace(t, '', 1)
                            partype = t.strip().strip(':').strip()
                            found = True
                            break
                    if not found:  #don't know the type listed in the help; default to string
                        partype = str
                else:
                    parhelp = ''
                    partype = str
                parhelp = parhelp.strip().strip('\n')
                if parhelp == ':':
                    parhelp = ''
                self.typedict[arg] = partype
                self.helpdict[arg] = parhelp

                #now replace the type with the type of the default, if there is one and it's not explicitely given
                if self.defaults[ii] != 'nodef' and not found: 
                    self.typedict[arg] = type(self.defaults[ii])

            # self.parser = argparse.ArgumentParser(description=self.docstring, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            self.parser = argparse.ArgumentParser(description=self.docstring, formatter_class=CustomHelpFormatter)
            
            for ii, arg in enumerate(self.args):
                h = self.helpdict[arg]
                try:
                    t = tdict[self.typedict[arg]]
                except KeyError:
                    t = str
                if self.defaults[ii] == 'nodef':
                    self.parser.add_argument(arg,type=t,help=h)
                else:
                    if h == '':
                        h = 'sets '+arg
                    self.parser.add_argument('--'+arg,type=t,help=h,default=self.defaults[ii])

    def parse_args(self):
        return vars(self.parser.parse_args())
