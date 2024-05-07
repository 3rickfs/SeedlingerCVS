let SessionLoad = 1
if &cp | set nocp | endif
let s:cpo_save=&cpo
set cpo&vim
imap <Nul> <C-Space>
inoremap <expr> <Up> pumvisible() ? "\" : "\<Up>"
inoremap <expr> <S-Tab> pumvisible() ? "\" : "\<S-Tab>"
inoremap <expr> <Down> pumvisible() ? "\" : "\<Down>"
inoremap <silent> <F3> :MaximizerToggle
imap <F5> <Plug>ToggleBackground
nnoremap  
nnoremap <NL> <NL>
nnoremap  
nnoremap  
nnoremap <silent>  :CtrlP
nnoremap  :terminal
noremap m :MaximizerToggle
nnoremap   za
nnoremap \d :YcmShowDetailedDiagnostic
map \g :YcmCompleter GoToDefinitionElseDeclaration
vmap gx <Plug>NetrwBrowseXVis
nmap gx <Plug>NetrwBrowseX
nnoremap py :!python %
nnoremap <silent> <Plug>(YCMFindSymbolInDocument) :call youcompleteme#finder#FindSymbol( 'document' )
nnoremap <silent> <Plug>(YCMFindSymbolInWorkspace) :call youcompleteme#finder#FindSymbol( 'workspace' )
vnoremap <silent> <Plug>NetrwBrowseXVis :call netrw#BrowseXVis()
nnoremap <silent> <Plug>NetrwBrowseX :call netrw#BrowseX(expand((exists("g:netrw_gx")? g:netrw_gx : '<cfile>')),netrw#CheckIfRemote())
vnoremap <silent> <F3> :MaximizerTogglegv
nnoremap <silent> <F3> :MaximizerToggle
nnoremap <silent> <C-P> :CtrlP
nmap <M-Up> :resize -1
nmap <M-Down> :resize +1
nmap <M-Left> :vertical resize -1
nmap <M-Right> :vertical resize +1
nnoremap <C-T> :terminal
noremap <C-W>m :MaximizerToggle
vmap <F5> <Plug>ToggleBackground
nmap <F5> <Plug>ToggleBackground
nnoremap <C-H> 
nnoremap <C-L> 
nnoremap <C-K> 
nnoremap <C-J> <NL>
inoremap <expr> 	 pumvisible() ? "\" : "\	"
inoremap jk 
let &cpo=s:cpo_save
unlet s:cpo_save
set background=dark
set backspace=indent,eol,start
set completeopt=preview,menuone
set cpoptions=aAceFsB
set fileencodings=ucs-bom,utf-8,default,latin1
set helplang=en
set printoptions=paper:a4
set ruler
set runtimepath=~/.vim,~/.vim/bundle/Vundle.vim,~/.vim/bundle/SimpylFold,~/.vim/bundle/indentpython.vim,~/.vim/bundle/YouCompleteMe,~/.vim/bundle/syntastic,~/.vim/bundle/vim-flake8,~/.vim/bundle/Zenburn,~/.vim/bundle/vim-colors-solarized,~/.vim/bundle/nerdtree,~/.vim/bundle/vim-nerdtree-tabs,~/.vim/bundle/ctrlp.vim,~/.vim/bundle/vim-fugitive,~/.vim/bundle/powerline/powerline/bindings/vim/,~/.vim/bundle/vim-maximizer,/var/lib/vim/addons,/etc/vim,/usr/share/vim/vimfiles,/usr/share/vim/vim81,/usr/share/vim/vimfiles/after,/etc/vim/after,/var/lib/vim/addons/after,~/.vim/after,~/.vim/bundle/Vundle.vim/after,~/.vim/bundle/SimpylFold/after,~/.vim/bundle/indentpython.vim/after,~/.vim/bundle/YouCompleteMe/after,~/.vim/bundle/syntastic/after,~/.vim/bundle/vim-flake8/after,~/.vim/bundle/Zenburn/after,~/.vim/bundle/vim-colors-solarized/after,~/.vim/bundle/nerdtree/after,~/.vim/bundle/vim-nerdtree-tabs/after,~/.vim/bundle/ctrlp.vim/after,~/.vim/bundle/vim-fugitive/after,~/.vim/bundle/powerline/powerline/bindings/vim//after,~/.vim/bundle/vim-maximizer/after
set shortmess=filnxtToOSc
set splitbelow
set splitright
set statusline=%!py3eval('powerline.new_window()')
set suffixes=.bak,~,.swp,.o,.info,.aux,.log,.dvi,.bbl,.blg,.brf,.cb,.ind,.idx,.ilg,.inx,.out,.toc
set tabline=%!py3eval('powerline.tabline()')
set tabstop=4
set wildignore=*.pyc
set window=52
let s:so_save = &so | let s:siso_save = &siso | set so=0 siso=0
let v:this_session=expand("<sfile>:p")
silent only
silent tabonly
cd ~/dev/SeedlingerRobotCVSytem
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
set shortmess=aoO
argglobal
%argdel
$argadd seedlinger_cvs.py
edit seedlinger_cvs.py
set splitbelow splitright
wincmd t
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
argglobal
let s:cpo_save=&cpo
set cpo&vim
noremap <buffer> <F7> :call flake8#Flake8()
let &cpo=s:cpo_save
unlet s:cpo_save
setlocal keymap=
setlocal noarabic
setlocal autoindent
setlocal backupcopy=
setlocal balloonexpr=
setlocal nobinary
setlocal nobreakindent
setlocal breakindentopt=
setlocal bufhidden=
setlocal buflisted
setlocal buftype=
setlocal nocindent
setlocal cinkeys=0{,0},0),0],:,!^F,o,O,e
setlocal cinoptions=
setlocal cinwords=if,else,while,do,for,switch
setlocal colorcolumn=
setlocal comments=b:#,fb:-
setlocal commentstring=#\ %s
setlocal complete=.,w,b,u,t,i
setlocal concealcursor=
setlocal conceallevel=0
setlocal completefunc=
setlocal nocopyindent
setlocal cryptmethod=
setlocal nocursorbind
setlocal nocursorcolumn
setlocal nocursorline
setlocal cursorlineopt=both
setlocal define=
setlocal dictionary=
setlocal nodiff
setlocal equalprg=
setlocal errorformat=
setlocal expandtab
if &filetype != 'python'
setlocal filetype=python
endif
setlocal fixendofline
setlocal foldcolumn=0
setlocal foldenable
setlocal foldexpr=SimpylFold#FoldExpr(v:lnum)
setlocal foldignore=#
set foldlevel=99
setlocal foldlevel=99
setlocal foldmarker={{{,}}}
set foldmethod=indent
setlocal foldmethod=expr
setlocal foldminlines=1
setlocal foldnestmax=20
setlocal foldtext=foldtext()\ .\ SimpylFold#FoldText()
setlocal formatexpr=
setlocal formatoptions=tcq
setlocal formatlistpat=^\\s*\\d\\+[\\]:.)}\\t\ ]\\s*
setlocal formatprg=
setlocal grepprg=
setlocal iminsert=0
setlocal imsearch=-1
setlocal include=^\\s*\\(from\\|import\\)
setlocal includeexpr=substitute(substitute(substitute(v:fname,b:grandparent_match,b:grandparent_sub,''),b:parent_match,b:parent_sub,''),b:child_match,b:child_sub,'g')
setlocal indentexpr=GetPythonIndent(v:lnum)
setlocal indentkeys=!^F,o,O,<:>,0),0],0},=elif,=except
setlocal noinfercase
setlocal iskeyword=@,48-57,_,192-255
setlocal keywordprg=
setlocal nolinebreak
setlocal nolisp
setlocal lispwords=
setlocal nolist
setlocal makeencoding=
setlocal makeprg=
setlocal matchpairs=(:),{:},[:]
setlocal modeline
setlocal modifiable
setlocal nrformats=bin,octal,hex
set number
setlocal number
setlocal numberwidth=4
setlocal omnifunc=python3complete#Complete
setlocal path=
setlocal nopreserveindent
setlocal nopreviewwindow
setlocal quoteescape=\\
setlocal noreadonly
setlocal norelativenumber
setlocal norightleft
setlocal rightleftcmd=search
setlocal noscrollbind
setlocal scrolloff=-1
setlocal shiftwidth=4
setlocal noshortname
setlocal sidescrolloff=-1
setlocal signcolumn=auto
setlocal nosmartindent
setlocal softtabstop=4
setlocal nospell
setlocal spellcapcheck=[.?!]\\_[\\])'\"\	\ ]\\+
setlocal spellfile=
setlocal spelllang=en
setlocal statusline=%!py3eval('powerline.statusline(1)')
setlocal suffixesadd=.py
setlocal swapfile
setlocal synmaxcol=3000
if &syntax != 'python'
setlocal syntax=python
endif
setlocal tabstop=4
setlocal tagcase=
setlocal tagfunc=
setlocal tags=
setlocal termwinkey=
setlocal termwinscroll=10000
setlocal termwinsize=
setlocal textwidth=0
setlocal thesaurus=
setlocal noundofile
setlocal undolevels=-123456
setlocal varsofttabstop=
setlocal vartabstop=
setlocal wincolor=
setlocal nowinfixheight
setlocal nowinfixwidth
setlocal wrap
setlocal wrapmargin=0
let s:l = 2 - ((1 * winheight(0) + 26) / 52)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
2
normal! 0
tabnext 1
badd +0 seedlinger_cvs.py
badd +26 ~/dev/marks_modbus_scripts/marks_modbus/MODBUS\ SERVER\ BRAZO/control_brazo.py
badd +36 ~/dev/marks_modbus_scripts/marks_modbus/MODBUS\ SERVER\ BRAZO/servidor_modbus_mp.py
badd +0 ~/dev/marks_modbus_scripts/marks_modbus/MODBUS\ SERVER\ BRAZO/src/MODBUS/entorno_1br.py
badd +65 ~/dev/marks_modbus_scripts/marks_modbus/MODBUS\ SERVER\ BRAZO/Control_BR.py
badd +96 ~/dev/marks_modbus_scripts/marks_modbus/MODBUS\ SERVER\ BRAZO/src/MODBUS/config_registros.py
if exists('s:wipebuf') && len(win_findbuf(s:wipebuf)) == 0
  silent exe 'bwipe ' . s:wipebuf
endif
unlet! s:wipebuf
set winheight=1 winwidth=20 shortmess=filnxtToOSc
set winminheight=1 winminwidth=1
let s:sx = expand("<sfile>:p:r")."x.vim"
if file_readable(s:sx)
  exe "source " . fnameescape(s:sx)
endif
let &so = s:so_save | let &siso = s:siso_save
doautoall SessionLoadPost
unlet SessionLoad
" vim: set ft=vim :
