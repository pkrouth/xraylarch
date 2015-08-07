#!/usr/bin/env python
"""
  Larch show() function
"""
import os
import sys
import types
import numpy
from larch import Group, ValidateLarchPlugin


HAS_TERMCOLOR = False
try:
    from termcolor import colored
    if os.name == 'nt':
        import colorama
        colorama.init(convert=True)
    HAS_TERMCOLOR = True
except:
    pass

@ValidateLarchPlugin
def _get(sym=None, _larch=None, **kws):
    """get object from symbol table from symbol name:

    >>> g = group(a = 1,  b=2.3, z = 'a string')
    >>> print get('g.z')
    'a string'

    """
    if sym is None:
        sym = '_main'
    group = None
    symtable = _larch.symtable
    if symtable.isgroup(sym):
        group = sym
    elif isinstance(sym, types.ModuleType):
        group = sym
    elif isinstance(sym, (str, unicode)):
        group = symtable._lookup(sym, create=False)
    return group

@ValidateLarchPlugin
def _show_old(sym=None, _larch=None, with_private=False, **kws):
    """display group members.
    Options
    -------
    with_private:  show 'private' members ('__private__')

    See Also:  show_tree()
    """
    if sym is None:
        sym = '_main'
    group = None
    symtable = _larch.symtable
    title = sym
    if symtable.isgroup(sym):
        group = sym
        title = repr(sym)[1:-1]
    elif isinstance(sym, types.ModuleType):
        group = sym
        title = sym.__name__

    if group is None:
        _larch.writer.write("%s\n" % repr(sym))
        return
    if title.startswith(symtable.top_group):
        title = title[6:]

    if group == symtable:
        title = 'Group _main'

    members = dir(group)
    out = ['== %s: %i symbols ==' % (title, len(members))]
    for item in members:
        if item.startswith('__') and not with_private:
		#item.endswith('__') and# not with_private):
            continue
        obj = getattr(group, item)
        dval = None
        if isinstance(obj, numpy.ndarray):
            if len(obj) > 10 or len(obj.shape)>1:
                dval = "array<shape=%s, type=%s>" % (repr(obj.shape),
                                                         repr(obj.dtype))
        if dval is None:
            dval = repr(obj)
        out.append('  %s: %s' % (item, dval))
#         if not (item.startswith('_Group__') or
#                 item == '__name__' or item == '_larch' or
#                 item.startswith('_SymbolTable__')):

    _larch.writer.write("%s\n" % '\n'.join(out))

@ValidateLarchPlugin
def show_tree(group, _larch=None, indent=0, groups_shown=None, **kws):
    """show members of a Group, with a tree structure for sub-groups

    > show_tree(group1)

    """
    if groups_shown is None:
        groups_shown = []
    for item in dir(group):
        if (item.startswith('__') and item.endswith('__')):
            continue
        obj = getattr(group, item)
        dval = None
        if _larch.symtable.isgroup(obj):
            _larch.writer.write('%s %s: %s\n' % (indent*' ', item, obj))
            if id(obj) in groups_shown:
                _larch.writer.write('%s     (shown above)\n' % (indent*' '))
            else:
                groups_shown.append(id(obj))
                show_tree(obj, indent=indent+3, _larch=_larch, groups_shown=groups_shown)
        else:
            dval = repr(obj)
            if isinstance(obj, numpy.ndarray):
                if len(obj) > 10 or len(obj.shape)>1:
                    dval = "array<shape=%s, type=%s>" % (repr(obj.shape),
                                                         repr(obj.dtype))
            _larch.writer.write('%s %s: %s\n' % (indent*' ', item, dval))

def group2dict(group, _larch=None):
    "return dictionary of group members"
    return group.__dict__

def dict2group(d, _larch=None):
    "return group created from a dictionary"
    return Group(**d)



@ValidateLarchPlugin
def _show(sym=None, _larch=None, with_private=False, with_color=True, 
          color='cyan', truncate=True, with_methods=True, **kws):
    """show group members:
    Options
    -------
    with_private:  show 'private' members ('__private__') if True
    with_color:    show alternating lines in color if True and color is available.
    truncate:      truncate representation of lengthy lists and tuples if True
    with_methods:  suppress display of methods if False

    """
    with_color = with_color and HAS_TERMCOLOR

    if sym is None:
        sym = '_main'
    group = None
    symtable = _larch.symtable
    title = sym
    if symtable.isgroup(sym):
        group = sym
        title = repr(sym)[1:-1]
    elif isinstance(sym, types.ModuleType):
        group = sym
        title = sym.__name__

    if group is None:
        _larch.writer.write("%s\n" % repr(sym))
        return
    if title.startswith(symtable.top_group):
        title = title[6:]

    if group == symtable:
        title = 'Group _main'

    ## these are the 8 allowed colors in termcolor
    if (not color in ('grey', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white')):
        color = 'cyan'

    members = dir(group)
    out = ['== %s: %i symbols ==' % (title, len(members))]
    count = 0
    for item in members:
        if (item.startswith('__') and item.endswith('__') and
            not with_private):
            continue
        obj = getattr(group, item)
        if (callable(obj) and not with_methods):
            continue
        count = count+1
        dval = None
        if isinstance(obj, numpy.ndarray):
            if len(obj) > 10 or len(obj.shape)>1:
                dval = "array<shape=%s, type=%s>" % (repr(obj.shape),
                                                         repr(obj.dtype))
        if ((isinstance(obj, list) or isinstance(obj, tuple)) and truncate):
            if len(repr(obj)) > 50:
                dval = "[%s, %s, ... %s, %s]" % (repr(obj[0]), repr(obj[1]),
                                                 repr(obj[-2]), repr(obj[-1]))
        if dval is None:
            dval = repr(obj)
        if ((not with_color) or (count % 2)):
            string = '  %s: %s' % (item, dval)
        else:
            string = colored('  %s: %s' % (item, dval), color)
        out.append(string)

    if not with_methods:
        out[0] = '== %s: %i methods, %i attributes ==' % (title, len(members)-count, count)
    _larch.writer.write("%s\n" % '\n'.join(out))


def initializeLarchPlugin(_larch=None):
    """initialize show and friends"""
    cmds = ['show', 'show_tree']
    if _larch is not None:
        _larch.symtable._sys.valid_commands.extend(cmds)

def registerLarchPlugin():
    return ('_builtin', {'show': _show, 'get': _get,
                         'group2dict': group2dict,
                         'dict2group': dict2group,
                         'show_tree': show_tree, 
                         'show_simple': _show_old})
