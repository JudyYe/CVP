(function(){var h;function aa(a){var b=0;return function(){return b<a.length?{done:!1,value:a[b++]}:{done:!0}}}
function ba(a){var b="undefined"!=typeof Symbol&&Symbol.iterator&&a[Symbol.iterator];return b?b.call(a):{next:aa(a)}}
var ca="function"==typeof Object.create?Object.create:function(a){function b(){}
b.prototype=a;return new b},ea;
if("function"==typeof Object.setPrototypeOf)ea=Object.setPrototypeOf;else{var fa;a:{var ha={oa:!0},ia={};try{ia.__proto__=ha;fa=ia.oa;break a}catch(a){}fa=!1}ea=fa?function(a,b){a.__proto__=b;if(a.__proto__!==b)throw new TypeError(a+" is not extensible");return a}:null}var ja=ea;
function n(a,b){a.prototype=ca(b.prototype);a.prototype.constructor=a;if(ja)ja(a,b);else for(var c in b)if("prototype"!=c)if(Object.defineProperties){var d=Object.getOwnPropertyDescriptor(b,c);d&&Object.defineProperty(a,c,d)}else a[c]=b[c];a.w=b.prototype}
var ka="function"==typeof Object.defineProperties?Object.defineProperty:function(a,b,c){a!=Array.prototype&&a!=Object.prototype&&(a[b]=c.value)},la="undefined"!=typeof window&&window===this?this:"undefined"!=typeof global&&null!=global?global:this;
function ma(a,b){if(b){for(var c=la,d=a.split("."),e=0;e<d.length-1;e++){var f=d[e];f in c||(c[f]={});c=c[f]}d=d[d.length-1];e=c[d];f=b(e);f!=e&&null!=f&&ka(c,d,{configurable:!0,writable:!0,value:f})}}
function na(a,b,c){if(null==a)throw new TypeError("The 'this' value for String.prototype."+c+" must not be null or undefined");if(b instanceof RegExp)throw new TypeError("First argument to String.prototype."+c+" must not be a regular expression");return a+""}
function oa(){oa=function(){};
la.Symbol||(la.Symbol=pa)}
var pa=function(){var a=0;return function(b){return"jscomp_symbol_"+(b||"")+a++}}();
function qa(){oa();var a=la.Symbol.iterator;a||(a=la.Symbol.iterator=la.Symbol("iterator"));"function"!=typeof Array.prototype[a]&&ka(Array.prototype,a,{configurable:!0,writable:!0,value:function(){return ra(aa(this))}});
qa=function(){}}
function ra(a){qa();a={next:a};a[la.Symbol.iterator]=function(){return this};
return a}
ma("String.prototype.endsWith",function(a){return a?a:function(a,c){var b=na(this,a,"endsWith");a+="";void 0===c&&(c=b.length);for(var e=Math.max(0,Math.min(c|0,b.length)),f=a.length;0<f&&0<e;)if(b[--e]!=a[--f])return!1;return 0>=f}});
ma("String.prototype.startsWith",function(a){return a?a:function(a,c){var b=na(this,a,"startsWith");a+="";for(var e=b.length,f=a.length,g=Math.max(0,Math.min(c|0,b.length)),k=0;k<f&&g<e;)if(b[g++]!=a[k++])return!1;return k>=f}});
function p(a,b){return Object.prototype.hasOwnProperty.call(a,b)}
var sa="function"==typeof Object.assign?Object.assign:function(a,b){for(var c=1;c<arguments.length;c++){var d=arguments[c];if(d)for(var e in d)p(d,e)&&(a[e]=d[e])}return a};
ma("Object.assign",function(a){return a||sa});
ma("WeakMap",function(a){function b(a){this.b=(g+=Math.random()+1).toString();if(a){a=ba(a);for(var b;!(b=a.next()).done;)b=b.value,this.set(b[0],b[1])}}
function c(){}
function d(a){p(a,f)||ka(a,f,{value:new c})}
function e(a){var b=Object[a];b&&(Object[a]=function(a){if(a instanceof c)return a;d(a);return b(a)})}
if(function(){if(!a||!Object.seal)return!1;try{var b=Object.seal({}),c=Object.seal({}),d=new a([[b,2],[c,3]]);if(2!=d.get(b)||3!=d.get(c))return!1;d["delete"](b);d.set(c,4);return!d.has(b)&&4==d.get(c)}catch(u){return!1}}())return a;
var f="$jscomp_hidden_"+Math.random();e("freeze");e("preventExtensions");e("seal");var g=0;b.prototype.set=function(a,b){d(a);if(!p(a,f))throw Error("WeakMap key fail: "+a);a[f][this.b]=b;return this};
b.prototype.get=function(a){return p(a,f)?a[f][this.b]:void 0};
b.prototype.has=function(a){return p(a,f)&&p(a[f],this.b)};
b.prototype["delete"]=function(a){return p(a,f)&&p(a[f],this.b)?delete a[f][this.b]:!1};
return b});
ma("Map",function(a){function b(){var a={};return a.previous=a.next=a.head=a}
function c(a,b){var c=a.b;return ra(function(){if(c){for(;c.head!=a.b;)c=c.previous;for(;c.next!=c.head;)return c=c.next,{done:!1,value:b(c)};c=null}return{done:!0,value:void 0}})}
function d(a,b){var c=b&&typeof b;"object"==c||"function"==c?f.has(b)?c=f.get(b):(c=""+ ++g,f.set(b,c)):c="p_"+b;var d=a.c[c];if(d&&p(a.c,c))for(var e=0;e<d.length;e++){var k=d[e];if(b!==b&&k.key!==k.key||b===k.key)return{id:c,list:d,index:e,m:k}}return{id:c,list:d,index:-1,m:void 0}}
function e(a){this.c={};this.b=b();this.size=0;if(a){a=ba(a);for(var c;!(c=a.next()).done;)c=c.value,this.set(c[0],c[1])}}
if(function(){if(!a||"function"!=typeof a||!a.prototype.entries||"function"!=typeof Object.seal)return!1;try{var b=Object.seal({x:4}),c=new a(ba([[b,"s"]]));if("s"!=c.get(b)||1!=c.size||c.get({x:4})||c.set({x:4},"t")!=c||2!=c.size)return!1;var d=c.entries(),e=d.next();if(e.done||e.value[0]!=b||"s"!=e.value[1])return!1;e=d.next();return e.done||4!=e.value[0].x||"t"!=e.value[1]||!d.next().done?!1:!0}catch(da){return!1}}())return a;
qa();var f=new WeakMap;e.prototype.set=function(a,b){a=0===a?0:a;var c=d(this,a);c.list||(c.list=this.c[c.id]=[]);c.m?c.m.value=b:(c.m={next:this.b,previous:this.b.previous,head:this.b,key:a,value:b},c.list.push(c.m),this.b.previous.next=c.m,this.b.previous=c.m,this.size++);return this};
e.prototype["delete"]=function(a){a=d(this,a);return a.m&&a.list?(a.list.splice(a.index,1),a.list.length||delete this.c[a.id],a.m.previous.next=a.m.next,a.m.next.previous=a.m.previous,a.m.head=null,this.size--,!0):!1};
e.prototype.clear=function(){this.c={};this.b=this.b.previous=b();this.size=0};
e.prototype.has=function(a){return!!d(this,a).m};
e.prototype.get=function(a){return(a=d(this,a).m)&&a.value};
e.prototype.entries=function(){return c(this,function(a){return[a.key,a.value]})};
e.prototype.keys=function(){return c(this,function(a){return a.key})};
e.prototype.values=function(){return c(this,function(a){return a.value})};
e.prototype.forEach=function(a,b){for(var c=this.entries(),d;!(d=c.next()).done;)d=d.value,a.call(b,d[1],d[0],this)};
e.prototype[Symbol.iterator]=e.prototype.entries;var g=0;return e});
ma("Set",function(a){function b(a){this.b=new Map;if(a){a=ba(a);for(var b;!(b=a.next()).done;)this.add(b.value)}this.size=this.b.size}
if(function(){if(!a||"function"!=typeof a||!a.prototype.entries||"function"!=typeof Object.seal)return!1;try{var b=Object.seal({x:4}),d=new a(ba([b]));if(!d.has(b)||1!=d.size||d.add(b)!=d||1!=d.size||d.add({x:4})!=d||2!=d.size)return!1;var e=d.entries(),f=e.next();if(f.done||f.value[0]!=b||f.value[1]!=b)return!1;f=e.next();return f.done||f.value[0]==b||4!=f.value[0].x||f.value[1]!=f.value[0]?!1:e.next().done}catch(g){return!1}}())return a;
qa();b.prototype.add=function(a){a=0===a?0:a;this.b.set(a,a);this.size=this.b.size;return this};
b.prototype["delete"]=function(a){a=this.b["delete"](a);this.size=this.b.size;return a};
b.prototype.clear=function(){this.b.clear();this.size=0};
b.prototype.has=function(a){return this.b.has(a)};
b.prototype.entries=function(){return this.b.entries()};
b.prototype.values=function(){return this.b.values()};
b.prototype.keys=b.prototype.values;b.prototype[Symbol.iterator]=b.prototype.values;b.prototype.forEach=function(a,b){var c=this;this.b.forEach(function(d){return a.call(b,d,d,c)})};
return b});
ma("Object.is",function(a){return a?a:function(a,c){return a===c?0!==a||1/a===1/c:a!==a&&c!==c}});
ma("String.prototype.includes",function(a){return a?a:function(a,c){return-1!==na(this,a,"includes").indexOf(a,c||0)}});
(function(){function a(){function a(){}
Reflect.construct(a,[],function(){});
return new a instanceof a}
if("undefined"!=typeof Reflect&&Reflect.construct){if(a())return Reflect.construct;var b=Reflect.construct;return function(a,d,e){a=b(a,d);e&&Reflect.setPrototypeOf(a,e.prototype);return a}}return function(a,b,e){void 0===e&&(e=a);
e=ca(e.prototype||Object.prototype);return Function.prototype.apply.call(a,e,b)||e}})();
var q=this;function r(a){return void 0!==a}
function t(a){return"string"==typeof a}
function v(a,b,c){a=a.split(".");c=c||q;a[0]in c||"undefined"==typeof c.execScript||c.execScript("var "+a[0]);for(var d;a.length&&(d=a.shift());)!a.length&&r(b)?c[d]=b:c[d]&&c[d]!==Object.prototype[d]?c=c[d]:c=c[d]={}}
var ta=/^[\w+/_-]+[=]{0,2}$/,ua=null;function w(a,b){for(var c=a.split("."),d=b||q,e=0;e<c.length;e++)if(d=d[c[e]],null==d)return null;return d}
function va(){}
function wa(a){a.ba=void 0;a.getInstance=function(){return a.ba?a.ba:a.ba=new a}}
function xa(a){var b=typeof a;if("object"==b)if(a){if(a instanceof Array)return"array";if(a instanceof Object)return b;var c=Object.prototype.toString.call(a);if("[object Window]"==c)return"object";if("[object Array]"==c||"number"==typeof a.length&&"undefined"!=typeof a.splice&&"undefined"!=typeof a.propertyIsEnumerable&&!a.propertyIsEnumerable("splice"))return"array";if("[object Function]"==c||"undefined"!=typeof a.call&&"undefined"!=typeof a.propertyIsEnumerable&&!a.propertyIsEnumerable("call"))return"function"}else return"null";
else if("function"==b&&"undefined"==typeof a.call)return"object";return b}
function x(a){return"array"==xa(a)}
function ya(a){var b=xa(a);return"array"==b||"object"==b&&"number"==typeof a.length}
function za(a){return"function"==xa(a)}
function y(a){var b=typeof a;return"object"==b&&null!=a||"function"==b}
var Aa="closure_uid_"+(1E9*Math.random()>>>0),Ba=0;function Ca(a,b,c){return a.call.apply(a.bind,arguments)}
function Da(a,b,c){if(!a)throw Error();if(2<arguments.length){var d=Array.prototype.slice.call(arguments,2);return function(){var c=Array.prototype.slice.call(arguments);Array.prototype.unshift.apply(c,d);return a.apply(b,c)}}return function(){return a.apply(b,arguments)}}
function z(a,b,c){Function.prototype.bind&&-1!=Function.prototype.bind.toString().indexOf("native code")?z=Ca:z=Da;return z.apply(null,arguments)}
function Ea(a,b){var c=Array.prototype.slice.call(arguments,1);return function(){var b=c.slice();b.push.apply(b,arguments);return a.apply(this,b)}}
var A=Date.now||function(){return+new Date};
function B(a,b){v(a,b,void 0)}
function C(a,b){function c(){}
c.prototype=b.prototype;a.w=b.prototype;a.prototype=new c;a.prototype.constructor=a;a.kb=function(a,c,f){for(var d=Array(arguments.length-2),e=2;e<arguments.length;e++)d[e-2]=arguments[e];return b.prototype[c].apply(a,d)}}
;var Fa=document,D=window;function E(a){if(Error.captureStackTrace)Error.captureStackTrace(this,E);else{var b=Error().stack;b&&(this.stack=b)}a&&(this.message=String(a))}
C(E,Error);E.prototype.name="CustomError";var Ga=Array.prototype.indexOf?function(a,b){return Array.prototype.indexOf.call(a,b,void 0)}:function(a,b){if(t(a))return t(b)&&1==b.length?a.indexOf(b,0):-1;
for(var c=0;c<a.length;c++)if(c in a&&a[c]===b)return c;return-1},F=Array.prototype.forEach?function(a,b,c){Array.prototype.forEach.call(a,b,c)}:function(a,b,c){for(var d=a.length,e=t(a)?a.split(""):a,f=0;f<d;f++)f in e&&b.call(c,e[f],f,a)},Ha=Array.prototype.filter?function(a,b){return Array.prototype.filter.call(a,b,void 0)}:function(a,b){for(var c=a.length,d=[],e=0,f=t(a)?a.split(""):a,g=0;g<c;g++)if(g in f){var k=f[g];
b.call(void 0,k,g,a)&&(d[e++]=k)}return d},Ia=Array.prototype.map?function(a,b){return Array.prototype.map.call(a,b,void 0)}:function(a,b){for(var c=a.length,d=Array(c),e=t(a)?a.split(""):a,f=0;f<c;f++)f in e&&(d[f]=b.call(void 0,e[f],f,a));
return d},Ja=Array.prototype.reduce?function(a,b,c){return Array.prototype.reduce.call(a,b,c)}:function(a,b,c){var d=c;
F(a,function(c,f){d=b.call(void 0,d,c,f,a)});
return d};
function Ka(a,b){a:{var c=a.length;for(var d=t(a)?a.split(""):a,e=0;e<c;e++)if(e in d&&b.call(void 0,d[e],e,a)){c=e;break a}c=-1}return 0>c?null:t(a)?a.charAt(c):a[c]}
function La(a,b){var c=Ga(a,b);0<=c&&Array.prototype.splice.call(a,c,1)}
function Ma(a){var b=a.length;if(0<b){for(var c=Array(b),d=0;d<b;d++)c[d]=a[d];return c}return[]}
function Na(a,b){for(var c=1;c<arguments.length;c++){var d=arguments[c];if(ya(d)){var e=a.length||0,f=d.length||0;a.length=e+f;for(var g=0;g<f;g++)a[e+g]=d[g]}else a.push(d)}}
;var Oa=String.prototype.trim?function(a){return a.trim()}:function(a){return/^[\s\xa0]*([\s\S]*?)[\s\xa0]*$/.exec(a)[1]};
function Pa(a){if(!Qa.test(a))return a;-1!=a.indexOf("&")&&(a=a.replace(Ra,"&amp;"));-1!=a.indexOf("<")&&(a=a.replace(Sa,"&lt;"));-1!=a.indexOf(">")&&(a=a.replace(Ta,"&gt;"));-1!=a.indexOf('"')&&(a=a.replace(Ua,"&quot;"));-1!=a.indexOf("'")&&(a=a.replace(Va,"&#39;"));-1!=a.indexOf("\x00")&&(a=a.replace(Wa,"&#0;"));return a}
var Ra=/&/g,Sa=/</g,Ta=/>/g,Ua=/"/g,Va=/'/g,Wa=/\x00/g,Qa=/[\x00&<>"']/;function Xa(a){for(var b=0,c=0;c<a.length;++c)b=31*b+a.charCodeAt(c)>>>0;return b}
;var Ya;a:{var Za=q.navigator;if(Za){var $a=Za.userAgent;if($a){Ya=$a;break a}}Ya=""}function G(a){return-1!=Ya.indexOf(a)}
;function ab(a,b){for(var c in a)b.call(void 0,a[c],c,a)}
function bb(a,b){var c=ya(b),d=c?b:arguments;for(c=c?0:1;c<d.length;c++){if(null==a)return;a=a[d[c]]}return a}
function cb(a){var b=db,c;for(c in b)if(a.call(void 0,b[c],c,b))return c}
function eb(a){for(var b in a)return!1;return!0}
function fb(a,b){if(null!==a&&b in a)throw Error('The object already contains the key "'+b+'"');a[b]=!0}
function gb(a,b){for(var c in a)if(!(c in b)||a[c]!==b[c])return!1;for(c in b)if(!(c in a))return!1;return!0}
function hb(a){var b={},c;for(c in a)b[c]=a[c];return b}
var ib="constructor hasOwnProperty isPrototypeOf propertyIsEnumerable toLocaleString toString valueOf".split(" ");function jb(a,b){for(var c,d,e=1;e<arguments.length;e++){d=arguments[e];for(c in d)a[c]=d[c];for(var f=0;f<ib.length;f++)c=ib[f],Object.prototype.hasOwnProperty.call(d,c)&&(a[c]=d[c])}}
;function kb(a){kb[" "](a);return a}
kb[" "]=va;var lb=G("Opera"),mb=G("Trident")||G("MSIE"),nb=G("Edge"),ob=G("Gecko")&&!(-1!=Ya.toLowerCase().indexOf("webkit")&&!G("Edge"))&&!(G("Trident")||G("MSIE"))&&!G("Edge"),pb=-1!=Ya.toLowerCase().indexOf("webkit")&&!G("Edge");function qb(){var a=q.document;return a?a.documentMode:void 0}
var rb;a:{var sb="",tb=function(){var a=Ya;if(ob)return/rv:([^\);]+)(\)|;)/.exec(a);if(nb)return/Edge\/([\d\.]+)/.exec(a);if(mb)return/\b(?:MSIE|rv)[: ]([^\);]+)(\)|;)/.exec(a);if(pb)return/WebKit\/(\S+)/.exec(a);if(lb)return/(?:Version)[ \/]?(\S+)/.exec(a)}();
tb&&(sb=tb?tb[1]:"");if(mb){var ub=qb();if(null!=ub&&ub>parseFloat(sb)){rb=String(ub);break a}}rb=sb}var vb=rb,wb;var xb=q.document;wb=xb&&mb?qb()||("CSS1Compat"==xb.compatMode?parseInt(vb,10):5):void 0;var yb=null,zb=null;function Ab(a){this.b=a||{cookie:""}}
h=Ab.prototype;h.isEnabled=function(){return navigator.cookieEnabled};
h.set=function(a,b,c,d,e,f){if(/[;=\s]/.test(a))throw Error('Invalid cookie name "'+a+'"');if(/[;\r\n]/.test(b))throw Error('Invalid cookie value "'+b+'"');r(c)||(c=-1);e=e?";domain="+e:"";d=d?";path="+d:"";f=f?";secure":"";c=0>c?"":0==c?";expires="+(new Date(1970,1,1)).toUTCString():";expires="+(new Date(A()+1E3*c)).toUTCString();this.b.cookie=a+"="+b+e+d+c+f};
h.get=function(a,b){for(var c=a+"=",d=(this.b.cookie||"").split(";"),e=0,f;e<d.length;e++){f=Oa(d[e]);if(0==f.lastIndexOf(c,0))return f.substr(c.length);if(f==a)return""}return b};
h.remove=function(a,b,c){var d=r(this.get(a));this.set(a,"",0,b,c);return d};
h.isEmpty=function(){return!this.b.cookie};
h.clear=function(){for(var a=(this.b.cookie||"").split(";"),b=[],c=[],d,e,f=0;f<a.length;f++)e=Oa(a[f]),d=e.indexOf("="),-1==d?(b.push(""),c.push(e)):(b.push(e.substring(0,d)),c.push(e.substring(d+1)));for(a=b.length-1;0<=a;a--)this.remove(b[a])};
var Bb=new Ab("undefined"==typeof document?null:document);function Cb(a){var b=w("window.location.href");null==a&&(a='Unknown Error of type "null/undefined"');if(t(a))return{message:a,name:"Unknown error",lineNumber:"Not available",fileName:b,stack:"Not available"};var c=!1;try{var d=a.lineNumber||a.line||"Not available"}catch(f){d="Not available",c=!0}try{var e=a.fileName||a.filename||a.sourceURL||q.$googDebugFname||b}catch(f){e="Not available",c=!0}return!c&&a.lineNumber&&a.fileName&&a.stack&&a.message&&a.name?a:(b=a.message,null==b&&(a.constructor&&a.constructor instanceof
Function?(a.constructor.name?b=a.constructor.name:(b=a.constructor,Db[b]?b=Db[b]:(b=String(b),Db[b]||(c=/function\s+([^\(]+)/m.exec(b),Db[b]=c?c[1]:"[Anonymous]"),b=Db[b])),b='Unknown Error of type "'+b+'"'):b="Unknown Error of unknown type"),{message:b,name:a.name||"UnknownError",lineNumber:d,fileName:e,stack:a.stack||"Not available"})}
var Db={};function Eb(a){var b=!1,c;return function(){b||(c=a(),b=!0);return c}}
;var Fb=!mb||9<=Number(wb);function Gb(){this.b="";this.c=Hb}
Gb.prototype.I=!0;Gb.prototype.H=function(){return this.b};
Gb.prototype.aa=!0;Gb.prototype.X=function(){return 1};
function Ib(a){if(a instanceof Gb&&a.constructor===Gb&&a.c===Hb)return a.b;xa(a);return"type_error:TrustedResourceUrl"}
var Hb={};function H(){this.b="";this.c=Jb}
H.prototype.I=!0;H.prototype.H=function(){return this.b};
H.prototype.aa=!0;H.prototype.X=function(){return 1};
function Kb(a){if(a instanceof H&&a.constructor===H&&a.c===Jb)return a.b;xa(a);return"type_error:SafeUrl"}
var Lb=/^(?:(?:https?|mailto|ftp):|[^:/?#]*(?:[/?#]|$))/i;function Mb(a){if(a instanceof H)return a;a="object"==typeof a&&a.I?a.H():String(a);Lb.test(a)||(a="about:invalid#zClosurez");return Nb(a)}
var Jb={};function Nb(a){var b=new H;b.b=a;return b}
Nb("about:blank");function Ob(){this.b="";this.f=Pb;this.c=null}
Ob.prototype.aa=!0;Ob.prototype.X=function(){return this.c};
Ob.prototype.I=!0;Ob.prototype.H=function(){return this.b};
var Pb={};function Qb(a,b){var c=new Ob;c.b=a;c.c=b;return c}
Qb("<!DOCTYPE html>",0);Qb("",0);Qb("<br>",0);function Rb(a,b){var c=b instanceof H?b:Mb(b);a.href=Kb(c)}
function Sb(a,b){a.src=Ib(b);if(null===ua)b:{var c=q.document;if((c=c.querySelector&&c.querySelector("script[nonce]"))&&(c=c.nonce||c.getAttribute("nonce"))&&ta.test(c)){ua=c;break b}ua=""}c=ua;c&&a.setAttribute("nonce",c)}
;function Tb(a,b){this.x=r(a)?a:0;this.y=r(b)?b:0}
h=Tb.prototype;h.clone=function(){return new Tb(this.x,this.y)};
h.equals=function(a){return a instanceof Tb&&(this==a?!0:this&&a?this.x==a.x&&this.y==a.y:!1)};
h.ceil=function(){this.x=Math.ceil(this.x);this.y=Math.ceil(this.y);return this};
h.floor=function(){this.x=Math.floor(this.x);this.y=Math.floor(this.y);return this};
h.round=function(){this.x=Math.round(this.x);this.y=Math.round(this.y);return this};function Ub(a,b){this.width=a;this.height=b}
h=Ub.prototype;h.clone=function(){return new Ub(this.width,this.height)};
h.aspectRatio=function(){return this.width/this.height};
h.isEmpty=function(){return!(this.width*this.height)};
h.ceil=function(){this.width=Math.ceil(this.width);this.height=Math.ceil(this.height);return this};
h.floor=function(){this.width=Math.floor(this.width);this.height=Math.floor(this.height);return this};
h.round=function(){this.width=Math.round(this.width);this.height=Math.round(this.height);return this};function Vb(a){var b=document;return t(a)?b.getElementById(a):a}
function Wb(a,b){ab(b,function(b,d){b&&"object"==typeof b&&b.I&&(b=b.H());"style"==d?a.style.cssText=b:"class"==d?a.className=b:"for"==d?a.htmlFor=b:Xb.hasOwnProperty(d)?a.setAttribute(Xb[d],b):0==d.lastIndexOf("aria-",0)||0==d.lastIndexOf("data-",0)?a.setAttribute(d,b):a[d]=b})}
var Xb={cellpadding:"cellPadding",cellspacing:"cellSpacing",colspan:"colSpan",frameborder:"frameBorder",height:"height",maxlength:"maxLength",nonce:"nonce",role:"role",rowspan:"rowSpan",type:"type",usemap:"useMap",valign:"vAlign",width:"width"};
function Yb(a,b,c){var d=arguments,e=document,f=String(d[0]),g=d[1];if(!Fb&&g&&(g.name||g.type)){f=["<",f];g.name&&f.push(' name="',Pa(g.name),'"');if(g.type){f.push(' type="',Pa(g.type),'"');var k={};jb(k,g);delete k.type;g=k}f.push(">");f=f.join("")}f=e.createElement(f);g&&(t(g)?f.className=g:x(g)?f.className=g.join(" "):Wb(f,g));2<d.length&&Zb(e,f,d);return f}
function Zb(a,b,c){function d(c){c&&b.appendChild(t(c)?a.createTextNode(c):c)}
for(var e=2;e<c.length;e++){var f=c[e];!ya(f)||y(f)&&0<f.nodeType?d(f):F($b(f)?Ma(f):f,d)}}
function $b(a){if(a&&"number"==typeof a.length){if(y(a))return"function"==typeof a.item||"string"==typeof a.item;if(za(a))return"function"==typeof a.item}return!1}
function ac(a,b){for(var c=0;a;){if(b(a))return a;a=a.parentNode;c++}return null}
;function bc(a){cc();var b=new Gb;b.b=a;return b}
var cc=va;function dc(){var a=ec;try{var b;if(b=!!a&&null!=a.location.href)a:{try{kb(a.foo);b=!0;break a}catch(c){}b=!1}return b}catch(c){return!1}}
function fc(a){var b=gc;if(b)for(var c in b)Object.prototype.hasOwnProperty.call(b,c)&&a.call(void 0,b[c],c,b)}
function hc(){var a=[];fc(function(b){a.push(b)});
return a}
var gc={Wa:"allow-forms",Xa:"allow-modals",Ya:"allow-orientation-lock",Za:"allow-pointer-lock",ab:"allow-popups",bb:"allow-popups-to-escape-sandbox",cb:"allow-presentation",eb:"allow-same-origin",fb:"allow-scripts",gb:"allow-top-navigation",hb:"allow-top-navigation-by-user-activation"},ic=Eb(function(){return hc()});
function jc(){var a=document.createElement("IFRAME").sandbox,b=a&&a.supports;if(!b)return{};var c={};F(ic(),function(d){b.call(a,d)&&(c[d]=!0)});
return c}
;function kc(a){"number"==typeof a&&(a=Math.round(a)+"px");return a}
;var lc=!!window.google_async_iframe_id,ec=lc&&window.parent||window;var mc=/^(?:([^:/?#.]+):)?(?:\/\/(?:([^/?#]*)@)?([^/#?]*?)(?::([0-9]+))?(?=[/#?]|$))?([^?#]+)?(?:\?([^#]*))?(?:#([\s\S]*))?$/;function I(a){return a?decodeURI(a):a}
function J(a,b){return b.match(mc)[a]||null}
function nc(a,b,c){if(x(b))for(var d=0;d<b.length;d++)nc(a,String(b[d]),c);else null!=b&&c.push(a+(""===b?"":"="+encodeURIComponent(String(b))))}
function oc(a){var b=[],c;for(c in a)nc(c,a[c],b);return b.join("&")}
function pc(a,b){var c=oc(b);if(c){var d=a.indexOf("#");0>d&&(d=a.length);var e=a.indexOf("?");if(0>e||e>d){e=d;var f=""}else f=a.substring(e+1,d);d=[a.substr(0,e),f,a.substr(d)];e=d[1];d[1]=c?e?e+"&"+c:c:e;c=d[0]+(d[1]?"?"+d[1]:"")+d[2]}else c=a;return c}
var qc=/#|$/;function rc(a,b){var c=a.search(qc);a:{var d=0;for(var e=b.length;0<=(d=a.indexOf(b,d))&&d<c;){var f=a.charCodeAt(d-1);if(38==f||63==f)if(f=a.charCodeAt(d+e),!f||61==f||38==f||35==f)break a;d+=e+1}d=-1}if(0>d)return null;e=a.indexOf("&",d);if(0>e||e>c)e=c;d+=b.length+1;return decodeURIComponent(a.substr(d,e-d).replace(/\+/g," "))}
;var sc=null;function tc(){var a=q.performance;return a&&a.now&&a.timing?Math.floor(a.now()+a.timing.navigationStart):A()}
function uc(){var a=void 0===a?q:a;return(a=a.performance)&&a.now?a.now():null}
;function vc(a,b,c){this.label=a;this.type=b;this.value=c;this.duration=0;this.uniqueId=this.label+"_"+this.type+"_"+Math.random();this.slotId=void 0}
;var K=q.performance,wc=!!(K&&K.mark&&K.measure&&K.clearMarks),xc=Eb(function(){var a;if(a=wc){var b;if(null===sc){sc="";try{a="";try{a=q.top.location.hash}catch(c){a=q.location.hash}a&&(sc=(b=a.match(/\bdeid=([\d,]+)/))?b[1]:"")}catch(c){}}b=sc;a=!!b.indexOf&&0<=b.indexOf("1337")}return a});
function yc(){var a=zc;this.events=[];this.c=a||q;var b=null;a&&(a.google_js_reporting_queue=a.google_js_reporting_queue||[],this.events=a.google_js_reporting_queue,b=a.google_measure_js_timing);this.b=xc()||(null!=b?b:1>Math.random())}
yc.prototype.disable=function(){this.b=!1;this.events!=this.c.google_js_reporting_queue&&(xc()&&F(this.events,Ac),this.events.length=0)};
function Ac(a){a&&K&&xc()&&(K.clearMarks("goog_"+a.uniqueId+"_start"),K.clearMarks("goog_"+a.uniqueId+"_end"))}
yc.prototype.start=function(a,b){if(!this.b)return null;var c=uc()||tc();c=new vc(a,b,c);var d="goog_"+c.uniqueId+"_start";K&&xc()&&K.mark(d);return c};
yc.prototype.end=function(a){if(this.b&&"number"==typeof a.value){var b=uc()||tc();a.duration=b-a.value;b="goog_"+a.uniqueId+"_end";K&&xc()&&K.mark(b);this.b&&this.events.push(a)}};if(lc&&!dc()){var Bc="."+Fa.domain;try{for(;2<Bc.split(".").length&&!dc();)Fa.domain=Bc=Bc.substr(Bc.indexOf(".")+1),ec=window.parent}catch(a){}dc()||(ec=window)}var zc=ec,Cc=new yc;if("complete"==zc.document.readyState)zc.google_measure_js_timing||Cc.disable();else if(Cc.b){var Dc=function(){zc.google_measure_js_timing||Cc.disable()},Ec=zc;
Ec.addEventListener&&Ec.addEventListener("load",Dc,!1)};var Fc=(new Date).getTime();function Gc(a){if(!a)return"";a=a.split("#")[0].split("?")[0];a=a.toLowerCase();0==a.indexOf("//")&&(a=window.location.protocol+a);/^[\w\-]*:\/\//.test(a)||(a=window.location.href);var b=a.substring(a.indexOf("://")+3),c=b.indexOf("/");-1!=c&&(b=b.substring(0,c));a=a.substring(0,a.indexOf("://"));if("http"!==a&&"https"!==a&&"chrome-extension"!==a&&"file"!==a&&"android-app"!==a&&"chrome-search"!==a&&"app"!==a)throw Error("Invalid URI scheme in origin: "+a);c="";var d=b.indexOf(":");if(-1!=d){var e=
b.substring(d+1);b=b.substring(0,d);if("http"===a&&"80"!==e||"https"===a&&"443"!==e)c=":"+e}return a+"://"+b+c}
;function Hc(){function a(){e[0]=1732584193;e[1]=4023233417;e[2]=2562383102;e[3]=271733878;e[4]=3285377520;u=m=0}
function b(a){for(var b=g,c=0;64>c;c+=4)b[c/4]=a[c]<<24|a[c+1]<<16|a[c+2]<<8|a[c+3];for(c=16;80>c;c++)a=b[c-3]^b[c-8]^b[c-14]^b[c-16],b[c]=(a<<1|a>>>31)&4294967295;a=e[0];var d=e[1],f=e[2],k=e[3],l=e[4];for(c=0;80>c;c++){if(40>c)if(20>c){var m=k^d&(f^k);var u=1518500249}else m=d^f^k,u=1859775393;else 60>c?(m=d&f|k&(d|f),u=2400959708):(m=d^f^k,u=3395469782);m=((a<<5|a>>>27)&4294967295)+m+l+u+b[c]&4294967295;l=k;k=f;f=(d<<30|d>>>2)&4294967295;d=a;a=m}e[0]=e[0]+a&4294967295;e[1]=e[1]+d&4294967295;e[2]=
e[2]+f&4294967295;e[3]=e[3]+k&4294967295;e[4]=e[4]+l&4294967295}
function c(a,c){if("string"===typeof a){a=unescape(encodeURIComponent(a));for(var d=[],e=0,g=a.length;e<g;++e)d.push(a.charCodeAt(e));a=d}c||(c=a.length);d=0;if(0==m)for(;d+64<c;)b(a.slice(d,d+64)),d+=64,u+=64;for(;d<c;)if(f[m++]=a[d++],u++,64==m)for(m=0,b(f);d+64<c;)b(a.slice(d,d+64)),d+=64,u+=64}
function d(){var a=[],d=8*u;56>m?c(k,56-m):c(k,64-(m-56));for(var g=63;56<=g;g--)f[g]=d&255,d>>>=8;b(f);for(g=d=0;5>g;g++)for(var l=24;0<=l;l-=8)a[d++]=e[g]>>l&255;return a}
for(var e=[],f=[],g=[],k=[128],l=1;64>l;++l)k[l]=0;var m,u;a();return{reset:a,update:c,digest:d,ra:function(){for(var a=d(),b="",c=0;c<a.length;c++)b+="0123456789ABCDEF".charAt(Math.floor(a[c]/16))+"0123456789ABCDEF".charAt(a[c]%16);return b}}}
;function Ic(a,b,c){var d=[],e=[];if(1==(x(c)?2:1))return e=[b,a],F(d,function(a){e.push(a)}),Jc(e.join(" "));
var f=[],g=[];F(c,function(a){g.push(a.key);f.push(a.value)});
c=Math.floor((new Date).getTime()/1E3);e=0==f.length?[c,b,a]:[f.join(":"),c,b,a];F(d,function(a){e.push(a)});
a=Jc(e.join(" "));a=[c,a];0==g.length||a.push(g.join(""));return a.join("_")}
function Jc(a){var b=Hc();b.update(a);return b.ra().toLowerCase()}
;function Kc(a){var b=Gc(String(q.location.href)),c=q.__OVERRIDE_SID;null==c&&(c=(new Ab(document)).get("SID"));if(c&&(b=(c=0==b.indexOf("https:")||0==b.indexOf("chrome-extension:"))?q.__SAPISID:q.__APISID,null==b&&(b=(new Ab(document)).get(c?"SAPISID":"APISID")),b)){c=c?"SAPISIDHASH":"APISIDHASH";var d=String(q.location.href);return d&&b&&c?[c,Ic(Gc(d),b,a||null)].join(" "):null}return null}
;function Lc(a,b){this.f=a;this.g=b;this.c=0;this.b=null}
Lc.prototype.get=function(){if(0<this.c){this.c--;var a=this.b;this.b=a.next;a.next=null}else a=this.f();return a};
function Mc(a,b){a.g(b);100>a.c&&(a.c++,b.next=a.b,a.b=b)}
;function Nc(a){q.setTimeout(function(){throw a;},0)}
var Oc;
function Pc(){var a=q.MessageChannel;"undefined"===typeof a&&"undefined"!==typeof window&&window.postMessage&&window.addEventListener&&!G("Presto")&&(a=function(){var a=document.createElement("IFRAME");a.style.display="none";a.src="";document.documentElement.appendChild(a);var b=a.contentWindow;a=b.document;a.open();a.write("");a.close();var c="callImmediate"+Math.random(),d="file:"==b.location.protocol?"*":b.location.protocol+"//"+b.location.host;a=z(function(a){if(("*"==d||a.origin==d)&&a.data==
c)this.port1.onmessage()},this);
b.addEventListener("message",a,!1);this.port1={};this.port2={postMessage:function(){b.postMessage(c,d)}}});
if("undefined"!==typeof a&&!G("Trident")&&!G("MSIE")){var b=new a,c={},d=c;b.port1.onmessage=function(){if(r(c.next)){c=c.next;var a=c.fa;c.fa=null;a()}};
return function(a){d.next={fa:a};d=d.next;b.port2.postMessage(0)}}return"undefined"!==typeof document&&"onreadystatechange"in document.createElement("SCRIPT")?function(a){var b=document.createElement("SCRIPT");
b.onreadystatechange=function(){b.onreadystatechange=null;b.parentNode.removeChild(b);b=null;a();a=null};
document.documentElement.appendChild(b)}:function(a){q.setTimeout(a,0)}}
;function Qc(){this.c=this.b=null}
var Sc=new Lc(function(){return new Rc},function(a){a.reset()});
Qc.prototype.add=function(a,b){var c=Sc.get();c.set(a,b);this.c?this.c.next=c:this.b=c;this.c=c};
Qc.prototype.remove=function(){var a=null;this.b&&(a=this.b,this.b=this.b.next,this.b||(this.c=null),a.next=null);return a};
function Rc(){this.next=this.scope=this.b=null}
Rc.prototype.set=function(a,b){this.b=a;this.scope=b;this.next=null};
Rc.prototype.reset=function(){this.next=this.scope=this.b=null};function Tc(a,b){Uc||Vc();Wc||(Uc(),Wc=!0);Xc.add(a,b)}
var Uc;function Vc(){if(q.Promise&&q.Promise.resolve){var a=q.Promise.resolve(void 0);Uc=function(){a.then(Yc)}}else Uc=function(){var a=Yc;
!za(q.setImmediate)||q.Window&&q.Window.prototype&&!G("Edge")&&q.Window.prototype.setImmediate==q.setImmediate?(Oc||(Oc=Pc()),Oc(a)):q.setImmediate(a)}}
var Wc=!1,Xc=new Qc;function Yc(){for(var a;a=Xc.remove();){try{a.b.call(a.scope)}catch(b){Nc(b)}Mc(Sc,a)}Wc=!1}
;function Zc(){this.c=-1}
;function $c(){this.c=64;this.b=[];this.i=[];this.o=[];this.g=[];this.g[0]=128;for(var a=1;a<this.c;++a)this.g[a]=0;this.h=this.f=0;this.reset()}
C($c,Zc);$c.prototype.reset=function(){this.b[0]=1732584193;this.b[1]=4023233417;this.b[2]=2562383102;this.b[3]=271733878;this.b[4]=3285377520;this.h=this.f=0};
function ad(a,b,c){c||(c=0);var d=a.o;if(t(b))for(var e=0;16>e;e++)d[e]=b.charCodeAt(c)<<24|b.charCodeAt(c+1)<<16|b.charCodeAt(c+2)<<8|b.charCodeAt(c+3),c+=4;else for(e=0;16>e;e++)d[e]=b[c]<<24|b[c+1]<<16|b[c+2]<<8|b[c+3],c+=4;for(e=16;80>e;e++){var f=d[e-3]^d[e-8]^d[e-14]^d[e-16];d[e]=(f<<1|f>>>31)&4294967295}b=a.b[0];c=a.b[1];var g=a.b[2],k=a.b[3],l=a.b[4];for(e=0;80>e;e++){if(40>e)if(20>e){f=k^c&(g^k);var m=1518500249}else f=c^g^k,m=1859775393;else 60>e?(f=c&g|k&(c|g),m=2400959708):(f=c^g^k,m=
3395469782);f=(b<<5|b>>>27)+f+l+m+d[e]&4294967295;l=k;k=g;g=(c<<30|c>>>2)&4294967295;c=b;b=f}a.b[0]=a.b[0]+b&4294967295;a.b[1]=a.b[1]+c&4294967295;a.b[2]=a.b[2]+g&4294967295;a.b[3]=a.b[3]+k&4294967295;a.b[4]=a.b[4]+l&4294967295}
$c.prototype.update=function(a,b){if(null!=a){r(b)||(b=a.length);for(var c=b-this.c,d=0,e=this.i,f=this.f;d<b;){if(0==f)for(;d<=c;)ad(this,a,d),d+=this.c;if(t(a))for(;d<b;){if(e[f]=a.charCodeAt(d),++f,++d,f==this.c){ad(this,e);f=0;break}}else for(;d<b;)if(e[f]=a[d],++f,++d,f==this.c){ad(this,e);f=0;break}}this.f=f;this.h+=b}};
$c.prototype.digest=function(){var a=[],b=8*this.h;56>this.f?this.update(this.g,56-this.f):this.update(this.g,this.c-(this.f-56));for(var c=this.c-1;56<=c;c--)this.i[c]=b&255,b/=256;ad(this,this.i);for(c=b=0;5>c;c++)for(var d=24;0<=d;d-=8)a[b]=this.b[c]>>d&255,++b;return a};function L(){this.c=this.c;this.o=this.o}
L.prototype.c=!1;L.prototype.dispose=function(){this.c||(this.c=!0,this.j())};
function bd(a,b){a.c?r(void 0)?b.call(void 0):b():(a.o||(a.o=[]),a.o.push(r(void 0)?z(b,void 0):b))}
L.prototype.j=function(){if(this.o)for(;this.o.length;)this.o.shift()()};
function cd(a){a&&"function"==typeof a.dispose&&a.dispose()}
function dd(a){for(var b=0,c=arguments.length;b<c;++b){var d=arguments[b];ya(d)?dd.apply(null,d):cd(d)}}
;function ed(a){if(a.classList)return a.classList;a=a.className;return t(a)&&a.match(/\S+/g)||[]}
function fd(a,b){if(a.classList)var c=a.classList.contains(b);else c=ed(a),c=0<=Ga(c,b);return c}
function gd(){var a=document.body;a.classList?a.classList.remove("inverted-hdpi"):fd(a,"inverted-hdpi")&&(a.className=Ha(ed(a),function(a){return"inverted-hdpi"!=a}).join(" "))}
;var hd="StopIteration"in q?q.StopIteration:{message:"StopIteration",stack:""};function id(){}
id.prototype.next=function(){throw hd;};
id.prototype.D=function(){return this};
function jd(a){if(a instanceof id)return a;if("function"==typeof a.D)return a.D(!1);if(ya(a)){var b=0,c=new id;c.next=function(){for(;;){if(b>=a.length)throw hd;if(b in a)return a[b++];b++}};
return c}throw Error("Not implemented");}
function kd(a,b){if(ya(a))try{F(a,b,void 0)}catch(c){if(c!==hd)throw c;}else{a=jd(a);try{for(;;)b.call(void 0,a.next(),void 0,a)}catch(c){if(c!==hd)throw c;}}}
function ld(a){if(ya(a))return Ma(a);a=jd(a);var b=[];kd(a,function(a){b.push(a)});
return b}
;function md(a,b){this.f={};this.b=[];this.g=this.c=0;var c=arguments.length;if(1<c){if(c%2)throw Error("Uneven number of arguments");for(var d=0;d<c;d+=2)this.set(arguments[d],arguments[d+1])}else if(a)if(a instanceof md)for(c=nd(a),d=0;d<c.length;d++)this.set(c[d],a.get(c[d]));else for(d in a)this.set(d,a[d])}
function nd(a){od(a);return a.b.concat()}
h=md.prototype;h.equals=function(a,b){if(this===a)return!0;if(this.c!=a.c)return!1;var c=b||pd;od(this);for(var d,e=0;d=this.b[e];e++)if(!c(this.get(d),a.get(d)))return!1;return!0};
function pd(a,b){return a===b}
h.isEmpty=function(){return 0==this.c};
h.clear=function(){this.f={};this.g=this.c=this.b.length=0};
h.remove=function(a){return Object.prototype.hasOwnProperty.call(this.f,a)?(delete this.f[a],this.c--,this.g++,this.b.length>2*this.c&&od(this),!0):!1};
function od(a){if(a.c!=a.b.length){for(var b=0,c=0;b<a.b.length;){var d=a.b[b];Object.prototype.hasOwnProperty.call(a.f,d)&&(a.b[c++]=d);b++}a.b.length=c}if(a.c!=a.b.length){var e={};for(c=b=0;b<a.b.length;)d=a.b[b],Object.prototype.hasOwnProperty.call(e,d)||(a.b[c++]=d,e[d]=1),b++;a.b.length=c}}
h.get=function(a,b){return Object.prototype.hasOwnProperty.call(this.f,a)?this.f[a]:b};
h.set=function(a,b){Object.prototype.hasOwnProperty.call(this.f,a)||(this.c++,this.b.push(a),this.g++);this.f[a]=b};
h.forEach=function(a,b){for(var c=nd(this),d=0;d<c.length;d++){var e=c[d],f=this.get(e);a.call(b,f,e,this)}};
h.clone=function(){return new md(this)};
h.D=function(a){od(this);var b=0,c=this.g,d=this,e=new id;e.next=function(){if(c!=d.g)throw Error("The map has changed since the iterator was created");if(b>=d.b.length)throw hd;var e=d.b[b++];return a?e:d.f[e]};
return e};function qd(a){var b=[];rd(new sd,a,b);return b.join("")}
function sd(){}
function rd(a,b,c){if(null==b)c.push("null");else{if("object"==typeof b){if(x(b)){var d=b;b=d.length;c.push("[");for(var e="",f=0;f<b;f++)c.push(e),rd(a,d[f],c),e=",";c.push("]");return}if(b instanceof String||b instanceof Number||b instanceof Boolean)b=b.valueOf();else{c.push("{");e="";for(d in b)Object.prototype.hasOwnProperty.call(b,d)&&(f=b[d],"function"!=typeof f&&(c.push(e),td(d,c),c.push(":"),rd(a,f,c),e=","));c.push("}");return}}switch(typeof b){case "string":td(b,c);break;case "number":c.push(isFinite(b)&&
!isNaN(b)?String(b):"null");break;case "boolean":c.push(String(b));break;case "function":c.push("null");break;default:throw Error("Unknown type: "+typeof b);}}}
var ud={'"':'\\"',"\\":"\\\\","/":"\\/","\b":"\\b","\f":"\\f","\n":"\\n","\r":"\\r","\t":"\\t","\x0B":"\\u000b"},vd=/\uffff/.test("\uffff")?/[\\"\x00-\x1f\x7f-\uffff]/g:/[\\"\x00-\x1f\x7f-\xff]/g;function td(a,b){b.push('"',a.replace(vd,function(a){var b=ud[a];b||(b="\\u"+(a.charCodeAt(0)|65536).toString(16).substr(1),ud[a]=b);return b}),'"')}
;function wd(a){if(!a)return!1;try{return!!a.$goog_Thenable}catch(b){return!1}}
;function M(a){this.b=0;this.o=void 0;this.g=this.c=this.f=null;this.h=this.i=!1;if(a!=va)try{var b=this;a.call(void 0,function(a){xd(b,2,a)},function(a){xd(b,3,a)})}catch(c){xd(this,3,c)}}
function yd(){this.next=this.context=this.onRejected=this.c=this.b=null;this.f=!1}
yd.prototype.reset=function(){this.context=this.onRejected=this.c=this.b=null;this.f=!1};
var zd=new Lc(function(){return new yd},function(a){a.reset()});
function Ad(a,b,c){var d=zd.get();d.c=a;d.onRejected=b;d.context=c;return d}
function Bd(a){return new M(function(b,c){c(a)})}
M.prototype.then=function(a,b,c){return Cd(this,za(a)?a:null,za(b)?b:null,c)};
M.prototype.$goog_Thenable=!0;function Ed(a,b){return Cd(a,null,b,void 0)}
M.prototype.cancel=function(a){0==this.b&&Tc(function(){var b=new Fd(a);Gd(this,b)},this)};
function Gd(a,b){if(0==a.b)if(a.f){var c=a.f;if(c.c){for(var d=0,e=null,f=null,g=c.c;g&&(g.f||(d++,g.b==a&&(e=g),!(e&&1<d)));g=g.next)e||(f=g);e&&(0==c.b&&1==d?Gd(c,b):(f?(d=f,d.next==c.g&&(c.g=d),d.next=d.next.next):Hd(c),Id(c,e,3,b)))}a.f=null}else xd(a,3,b)}
function Jd(a,b){a.c||2!=a.b&&3!=a.b||Kd(a);a.g?a.g.next=b:a.c=b;a.g=b}
function Cd(a,b,c,d){var e=Ad(null,null,null);e.b=new M(function(a,g){e.c=b?function(c){try{var e=b.call(d,c);a(e)}catch(m){g(m)}}:a;
e.onRejected=c?function(b){try{var e=c.call(d,b);!r(e)&&b instanceof Fd?g(b):a(e)}catch(m){g(m)}}:g});
e.b.f=a;Jd(a,e);return e.b}
M.prototype.u=function(a){this.b=0;xd(this,2,a)};
M.prototype.A=function(a){this.b=0;xd(this,3,a)};
function xd(a,b,c){if(0==a.b){a===c&&(b=3,c=new TypeError("Promise cannot resolve to itself"));a.b=1;a:{var d=c,e=a.u,f=a.A;if(d instanceof M){Jd(d,Ad(e||va,f||null,a));var g=!0}else if(wd(d))d.then(e,f,a),g=!0;else{if(y(d))try{var k=d.then;if(za(k)){Ld(d,k,e,f,a);g=!0;break a}}catch(l){f.call(a,l);g=!0;break a}g=!1}}g||(a.o=c,a.b=b,a.f=null,Kd(a),3!=b||c instanceof Fd||Md(a,c))}}
function Ld(a,b,c,d,e){function f(a){k||(k=!0,d.call(e,a))}
function g(a){k||(k=!0,c.call(e,a))}
var k=!1;try{b.call(a,g,f)}catch(l){f(l)}}
function Kd(a){a.i||(a.i=!0,Tc(a.l,a))}
function Hd(a){var b=null;a.c&&(b=a.c,a.c=b.next,b.next=null);a.c||(a.g=null);return b}
M.prototype.l=function(){for(var a;a=Hd(this);)Id(this,a,this.b,this.o);this.i=!1};
function Id(a,b,c,d){if(3==c&&b.onRejected&&!b.f)for(;a&&a.h;a=a.f)a.h=!1;if(b.b)b.b.f=null,Nd(b,c,d);else try{b.f?b.c.call(b.context):Nd(b,c,d)}catch(e){Od.call(null,e)}Mc(zd,b)}
function Nd(a,b,c){2==b?a.c.call(a.context,c):a.onRejected&&a.onRejected.call(a.context,c)}
function Md(a,b){a.h=!0;Tc(function(){a.h&&Od.call(null,b)})}
var Od=Nc;function Fd(a){E.call(this,a)}
C(Fd,E);Fd.prototype.name="cancel";function N(a){L.call(this);this.i=1;this.g=[];this.h=0;this.b=[];this.f={};this.l=!!a}
C(N,L);h=N.prototype;h.subscribe=function(a,b,c){var d=this.f[a];d||(d=this.f[a]=[]);var e=this.i;this.b[e]=a;this.b[e+1]=b;this.b[e+2]=c;this.i=e+3;d.push(e);return e};
function Pd(a,b,c,d){if(b=a.f[b]){var e=a.b;(b=Ka(b,function(a){return e[a+1]==c&&e[a+2]==d}))&&a.K(b)}}
h.K=function(a){var b=this.b[a];if(b){var c=this.f[b];0!=this.h?(this.g.push(a),this.b[a+1]=va):(c&&La(c,a),delete this.b[a],delete this.b[a+1],delete this.b[a+2])}return!!b};
h.J=function(a,b){var c=this.f[a];if(c){for(var d=Array(arguments.length-1),e=1,f=arguments.length;e<f;e++)d[e-1]=arguments[e];if(this.l)for(e=0;e<c.length;e++){var g=c[e];Qd(this.b[g+1],this.b[g+2],d)}else{this.h++;try{for(e=0,f=c.length;e<f;e++)g=c[e],this.b[g+1].apply(this.b[g+2],d)}finally{if(this.h--,0<this.g.length&&0==this.h)for(;c=this.g.pop();)this.K(c)}}return 0!=e}return!1};
function Qd(a,b,c){Tc(function(){a.apply(b,c)})}
h.clear=function(a){if(a){var b=this.f[a];b&&(F(b,this.K,this),delete this.f[a])}else this.b.length=0,this.f={}};
h.j=function(){N.w.j.call(this);this.clear();this.g.length=0};function Rd(a){this.b=a}
Rd.prototype.set=function(a,b){r(b)?this.b.set(a,qd(b)):this.b.remove(a)};
Rd.prototype.get=function(a){try{var b=this.b.get(a)}catch(c){return}if(null!==b)try{return JSON.parse(b)}catch(c){throw"Storage: Invalid value was encountered";}};
Rd.prototype.remove=function(a){this.b.remove(a)};function Sd(a){this.b=a}
C(Sd,Rd);function Td(a){this.data=a}
function Ud(a){return!r(a)||a instanceof Td?a:new Td(a)}
Sd.prototype.set=function(a,b){Sd.w.set.call(this,a,Ud(b))};
Sd.prototype.c=function(a){a=Sd.w.get.call(this,a);if(!r(a)||a instanceof Object)return a;throw"Storage: Invalid value was encountered";};
Sd.prototype.get=function(a){if(a=this.c(a)){if(a=a.data,!r(a))throw"Storage: Invalid value was encountered";}else a=void 0;return a};function Vd(a){this.b=a}
C(Vd,Sd);Vd.prototype.set=function(a,b,c){if(b=Ud(b)){if(c){if(c<A()){Vd.prototype.remove.call(this,a);return}b.expiration=c}b.creation=A()}Vd.w.set.call(this,a,b)};
Vd.prototype.c=function(a){var b=Vd.w.c.call(this,a);if(b){var c=b.creation,d=b.expiration;if(d&&d<A()||c&&c>A())Vd.prototype.remove.call(this,a);else return b}};function Wd(){}
;function Xd(){}
C(Xd,Wd);Xd.prototype.clear=function(){var a=ld(this.D(!0)),b=this;F(a,function(a){b.remove(a)})};function Yd(a){this.b=a}
C(Yd,Xd);h=Yd.prototype;h.isAvailable=function(){if(!this.b)return!1;try{return this.b.setItem("__sak","1"),this.b.removeItem("__sak"),!0}catch(a){return!1}};
h.set=function(a,b){try{this.b.setItem(a,b)}catch(c){if(0==this.b.length)throw"Storage mechanism: Storage disabled";throw"Storage mechanism: Quota exceeded";}};
h.get=function(a){a=this.b.getItem(a);if(!t(a)&&null!==a)throw"Storage mechanism: Invalid value was encountered";return a};
h.remove=function(a){this.b.removeItem(a)};
h.D=function(a){var b=0,c=this.b,d=new id;d.next=function(){if(b>=c.length)throw hd;var d=c.key(b++);if(a)return d;d=c.getItem(d);if(!t(d))throw"Storage mechanism: Invalid value was encountered";return d};
return d};
h.clear=function(){this.b.clear()};
h.key=function(a){return this.b.key(a)};function Zd(){var a=null;try{a=window.localStorage||null}catch(b){}this.b=a}
C(Zd,Yd);function $d(a,b){this.c=a;this.b=null;if(mb&&!(9<=Number(wb))){ae||(ae=new md);this.b=ae.get(a);this.b||(b?this.b=document.getElementById(b):(this.b=document.createElement("userdata"),this.b.addBehavior("#default#userData"),document.body.appendChild(this.b)),ae.set(a,this.b));try{this.b.load(this.c)}catch(c){this.b=null}}}
C($d,Xd);var be={".":".2E","!":".21","~":".7E","*":".2A","'":".27","(":".28",")":".29","%":"."},ae=null;function ce(a){return"_"+encodeURIComponent(a).replace(/[.!~*'()%]/g,function(a){return be[a]})}
h=$d.prototype;h.isAvailable=function(){return!!this.b};
h.set=function(a,b){this.b.setAttribute(ce(a),b);de(this)};
h.get=function(a){a=this.b.getAttribute(ce(a));if(!t(a)&&null!==a)throw"Storage mechanism: Invalid value was encountered";return a};
h.remove=function(a){this.b.removeAttribute(ce(a));de(this)};
h.D=function(a){var b=0,c=this.b.XMLDocument.documentElement.attributes,d=new id;d.next=function(){if(b>=c.length)throw hd;var d=c[b++];if(a)return decodeURIComponent(d.nodeName.replace(/\./g,"%")).substr(1);d=d.nodeValue;if(!t(d))throw"Storage mechanism: Invalid value was encountered";return d};
return d};
h.clear=function(){for(var a=this.b.XMLDocument.documentElement,b=a.attributes.length;0<b;b--)a.removeAttribute(a.attributes[b-1].nodeName);de(this)};
function de(a){try{a.b.save(a.c)}catch(b){throw"Storage mechanism: Quota exceeded";}}
;function ee(a,b){this.c=a;this.b=b+"::"}
C(ee,Xd);ee.prototype.set=function(a,b){this.c.set(this.b+a,b)};
ee.prototype.get=function(a){return this.c.get(this.b+a)};
ee.prototype.remove=function(a){this.c.remove(this.b+a)};
ee.prototype.D=function(a){var b=this.c.D(!0),c=this,d=new id;d.next=function(){for(var d=b.next();d.substr(0,c.b.length)!=c.b;)d=b.next();return a?d.substr(c.b.length):c.c.get(d)};
return d};function fe(){this.c=[];this.b=-1}
fe.prototype.set=function(a,b){b=void 0===b?!0:b;0<=a&&52>a&&0===a%1&&this.c[a]!=b&&(this.c[a]=b,this.b=-1)};
fe.prototype.get=function(a){return!!this.c[a]};
function ge(a){-1==a.b&&(a.b=Ja(a.c,function(a,c,d){return c?a+Math.pow(2,d):a},0));
return a.b}
;function he(a,b){if(1<b.length)a[b[0]]=b[1];else{var c=b[0],d;for(d in c)a[d]=c[d]}}
var O=window.performance&&window.performance.timing&&window.performance.now?function(){return window.performance.timing.navigationStart+window.performance.now()}:function(){return(new Date).getTime()};var ie=window.yt&&window.yt.config_||window.ytcfg&&window.ytcfg.data_||{};v("yt.config_",ie,void 0);function P(a){he(ie,arguments)}
function Q(a,b){return a in ie?ie[a]:b}
function je(a){return Q(a,void 0)}
function ke(){return Q("PLAYER_CONFIG",{})}
;function le(){var a=me;w("yt.ads.biscotti.getId_")||v("yt.ads.biscotti.getId_",a,void 0)}
function ne(a){v("yt.ads.biscotti.lastId_",a,void 0)}
;function oe(a){var b=pe;a=void 0===a?w("yt.ads.biscotti.lastId_")||"":a;b=Object.assign(qe(b),re(b));b.ca_type="image";a&&(b.bid=a);return b}
function qe(a){var b={};b.dt=Fc;b.flash="0";a:{try{var c=a.b.top.location.href}catch(f){a=2;break a}a=c?c===a.c.location.href?0:1:2}b=(b.frm=a,b);b.u_tz=-(new Date).getTimezoneOffset();var d=void 0===d?D:d;try{var e=d.history.length}catch(f){e=0}b.u_his=e;b.u_java=!!D.navigator&&"unknown"!==typeof D.navigator.javaEnabled&&!!D.navigator.javaEnabled&&D.navigator.javaEnabled();D.screen&&(b.u_h=D.screen.height,b.u_w=D.screen.width,b.u_ah=D.screen.availHeight,b.u_aw=D.screen.availWidth,b.u_cd=D.screen.colorDepth);
D.navigator&&D.navigator.plugins&&(b.u_nplug=D.navigator.plugins.length);D.navigator&&D.navigator.mimeTypes&&(b.u_nmime=D.navigator.mimeTypes.length);return b}
function re(a){var b=a.b;try{var c=b.screenX;var d=b.screenY}catch(da){}try{var e=b.outerWidth;var f=b.outerHeight}catch(da){}try{var g=b.innerWidth;var k=b.innerHeight}catch(da){}b=[b.screenLeft,b.screenTop,c,d,b.screen?b.screen.availWidth:void 0,b.screen?b.screen.availTop:void 0,e,f,g,k];c=a.b.top;try{var l=(c||window).document,m="CSS1Compat"==l.compatMode?l.documentElement:l.body;var u=(new Ub(m.clientWidth,m.clientHeight)).round()}catch(da){u=new Ub(-12245933,-12245933)}l=u;u={};m=new fe;q.SVGElement&&
q.document.createElementNS&&m.set(0);c=jc();c["allow-top-navigation-by-user-activation"]&&m.set(1);c["allow-popups-to-escape-sandbox"]&&m.set(2);q.crypto&&q.crypto.subtle&&m.set(3);m=ge(m);u.bc=m;u.bih=l.height;u.biw=l.width;u.brdim=b.join();a=a.c;return u.vis={visible:1,hidden:2,prerender:3,preview:4,unloaded:5}[a.visibilityState||a.webkitVisibilityState||a.mozVisibilityState||""]||0,u.wgl=!!D.WebGLRenderingContext,u}
var pe=new function(){var a=window.document;this.b=window;this.c=a};A();function se(a){return a&&window.yterr?function(){try{return a.apply(this,arguments)}catch(b){R(b)}}:a}
function R(a,b,c,d,e){var f=w("yt.logging.errors.log");f?f(a,b,c,d,e):(f=Q("ERRORS",[]),f.push([a,b,c,d,e]),P("ERRORS",f))}
function te(a){R(a,"WARNING",void 0,void 0,void 0)}
;function S(a){return Q("EXPERIMENT_FLAGS",{})[a]}
;var ue=r(XMLHttpRequest)?function(){return new XMLHttpRequest}:r(ActiveXObject)?function(){return new ActiveXObject("Microsoft.XMLHTTP")}:null;
function ve(){if(!ue)return null;var a=ue();return"open"in a?a:null}
function we(a){switch(a&&"status"in a?a.status:-1){case 200:case 201:case 202:case 203:case 204:case 205:case 206:case 304:return!0;default:return!1}}
;function T(a,b){za(a)&&(a=se(a));return window.setTimeout(a,b)}
function U(a){window.clearTimeout(a)}
;function xe(a){var b=[];ab(a,function(a,d){var c=encodeURIComponent(String(d)),f;x(a)?f=a:f=[a];F(f,function(a){""==a?b.push(c):b.push(c+"="+encodeURIComponent(String(a)))})});
return b.join("&")}
function ye(a){"?"==a.charAt(0)&&(a=a.substr(1));a=a.split("&");for(var b={},c=0,d=a.length;c<d;c++){var e=a[c].split("=");if(1==e.length&&e[0]||2==e.length){var f=decodeURIComponent((e[0]||"").replace(/\+/g," "));e=decodeURIComponent((e[1]||"").replace(/\+/g," "));f in b?x(b[f])?Na(b[f],e):b[f]=[b[f],e]:b[f]=e}}return b}
function ze(a,b,c){var d=a.split("#",2);a=d[0];d=1<d.length?"#"+d[1]:"";var e=a.split("?",2);a=e[0];e=ye(e[1]||"");for(var f in b)!c&&null!==e&&f in e||(e[f]=b[f]);return pc(a,e)+d}
;var Ae={Authorization:"AUTHORIZATION","X-Goog-Visitor-Id":"SANDBOXED_VISITOR_ID","X-YouTube-Client-Name":"INNERTUBE_CONTEXT_CLIENT_NAME","X-YouTube-Client-Version":"INNERTUBE_CONTEXT_CLIENT_VERSION","X-Youtube-Identity-Token":"ID_TOKEN","X-YouTube-Page-CL":"PAGE_CL","X-YouTube-Page-Label":"PAGE_BUILD_LABEL","X-YouTube-Variants-Checksum":"VARIANTS_CHECKSUM"},Be="app debugcss debugjs expflag force_ad_params force_viral_ad_response_params forced_experiments internalcountrycode internalipoverride absolute_experiments conditional_experiments sbb sr_bns_address".split(" "),
Ce=!1;
function De(a,b){b=void 0===b?{}:b;if(!c)var c=window.location.href;var d=J(1,a),e=I(J(3,a));d&&e?(d=c,c=a.match(mc),d=d.match(mc),c=c[3]==d[3]&&c[1]==d[1]&&c[4]==d[4]):c=e?I(J(3,c))==e&&(Number(J(4,c))||null)==(Number(J(4,a))||null):!0;d=!!S("web_ajax_ignore_global_headers_if_set");for(var f in Ae)e=Q(Ae[f]),!e||!c&&!Ee(a,f)||d&&void 0!==b[f]||(b[f]=e);if(c||Ee(a,"X-YouTube-Utc-Offset"))b["X-YouTube-Utc-Offset"]=-(new Date).getTimezoneOffset();S("pass_biscotti_id_in_header_ajax")&&(c||Ee(a,"X-YouTube-Ad-Signals"))&&
(f=oe(void 0),b["X-YouTube-Ad-Signals"]=xe(f));return b}
function Fe(a){var b=window.location.search,c=I(J(3,a)),d=I(J(5,a));d=(c=c&&c.endsWith("youtube.com"))&&d&&d.startsWith("/api/");if(!c||d)return a;var e=ye(b),f={};F(Be,function(a){e[a]&&(f[a]=e[a])});
return ze(a,f||{},!1)}
function Ee(a,b){var c=Q("CORS_HEADER_WHITELIST")||{},d=I(J(3,a));return d?(c=c[d])?0<=Ga(c,b):!1:!0}
function Ge(a,b){if(window.fetch&&"XML"!=b.format){var c={method:b.method||"GET",credentials:"same-origin"};b.headers&&(c.headers=b.headers);a=He(a,b);var d=Ie(a,b);d&&(c.body=d);b.withCredentials&&(c.credentials="include");var e=!1,f;fetch(a,c).then(function(a){if(!e){e=!0;f&&U(f);var c=a.ok,d=function(d){d=d||{};var e=b.context||q;c?b.onSuccess&&b.onSuccess.call(e,d,a):b.onError&&b.onError.call(e,d,a);b.da&&b.da.call(e,d,a)};
"JSON"==(b.format||"JSON")&&(c||400<=a.status&&500>a.status)?a.json().then(d,function(){d(null)}):d(null)}});
b.ha&&0<b.timeout&&(f=T(function(){e||(e=!0,U(f),b.ha.call(b.context||q))},b.timeout))}else Je(a,b)}
function Je(a,b){var c=b.format||"JSON";a=He(a,b);var d=Ie(a,b),e=!1,f,g=Ke(a,function(a){if(!e){e=!0;f&&U(f);var d=we(a),g=null,k=400<=a.status&&500>a.status,da=500<=a.status&&600>a.status;if(d||k||da)g=Le(c,a,b.mb);if(d)a:if(a&&204==a.status)d=!0;else{switch(c){case "XML":d=0==parseInt(g&&g.return_code,10);break a;case "RAW":d=!0;break a}d=!!g}g=g||{};k=b.context||q;d?b.onSuccess&&b.onSuccess.call(k,a,g):b.onError&&b.onError.call(k,a,g);b.da&&b.da.call(k,a,g)}},b.method,d,b.headers,b.responseType,
b.withCredentials);
b.L&&0<b.timeout&&(f=T(function(){e||(e=!0,g.abort(),U(f),b.L.call(b.context||q,g))},b.timeout));
return g}
function He(a,b){b.wa&&(a=document.location.protocol+"//"+document.location.hostname+(document.location.port?":"+document.location.port:"")+a);var c=Q("XSRF_FIELD_NAME",void 0),d=b.Va;d&&(d[c]&&delete d[c],a=ze(a,d||{},!0));return a}
function Ie(a,b){var c=Q("XSRF_FIELD_NAME",void 0),d=Q("XSRF_TOKEN",void 0),e=b.postBody||"",f=b.B,g=je("XSRF_FIELD_NAME"),k;b.headers&&(k=b.headers["Content-Type"]);b.nb||I(J(3,a))&&!b.withCredentials&&I(J(3,a))!=document.location.hostname||"POST"!=b.method||k&&"application/x-www-form-urlencoded"!=k||b.B&&b.B[g]||(f||(f={}),f[c]=d);f&&t(e)&&(e=ye(e),jb(e,f),e=b.ia&&"JSON"==b.ia?JSON.stringify(e):oc(e));f=e||f&&!eb(f);!Ce&&f&&"POST"!=b.method&&(Ce=!0,R(Error("AJAX request with postData should use POST")));
return e}
function Le(a,b,c){var d=null;switch(a){case "JSON":a=b.responseText;b=b.getResponseHeader("Content-Type")||"";a&&0<=b.indexOf("json")&&(d=JSON.parse(a));break;case "XML":if(b=(b=b.responseXML)?Me(b):null)d={},F(b.getElementsByTagName("*"),function(a){d[a.tagName]=Ne(a)})}c&&Oe(d);
return d}
function Oe(a){if(y(a))for(var b in a){var c;(c="html_content"==b)||(c=b.length-5,c=0<=c&&b.indexOf("_html",c)==c);if(c){c=b;var d=Qb(a[b],null);a[c]=d}else Oe(a[b])}}
function Me(a){return a?(a=("responseXML"in a?a.responseXML:a).getElementsByTagName("root"))&&0<a.length?a[0]:null:null}
function Ne(a){var b="";F(a.childNodes,function(a){b+=a.nodeValue});
return b}
function Pe(a,b){b.method="POST";b.B||(b.B={});Je(a,b)}
function Ke(a,b,c,d,e,f,g){function k(){4==(l&&"readyState"in l?l.readyState:0)&&b&&se(b)(l)}
c=void 0===c?"GET":c;d=void 0===d?"":d;var l=ve();if(!l)return null;"onloadend"in l?l.addEventListener("loadend",k,!1):l.onreadystatechange=k;S("debug_forward_web_query_parameters")&&(a=Fe(a));l.open(c,a,!0);f&&(l.responseType=f);g&&(l.withCredentials=!0);c="POST"==c&&(void 0===window.FormData||!(d instanceof FormData));if(e=De(a,e))for(var m in e)l.setRequestHeader(m,e[m]),"content-type"==m.toLowerCase()&&(c=!1);c&&l.setRequestHeader("Content-Type","application/x-www-form-urlencoded");l.send(d);
return l}
;var Qe={},Re=0;
function Se(a,b,c,d,e){e=void 0===e?"":e;a&&(c&&(c=Ya,c=!(c&&0<=c.toLowerCase().indexOf("cobalt"))),c?a&&(a instanceof H||(a="object"==typeof a&&a.I?a.H():String(a),Lb.test(a)||(a="about:invalid#zClosurez"),a=Nb(a)),b=Kb(a),"about:invalid#zClosurez"===b?a="":(b instanceof Ob?a=b:(d="object"==typeof b,a=null,d&&b.aa&&(a=b.X()),b=Pa(d&&b.I?b.H():String(b)),a=Qb(b,a)),a instanceof Ob&&a.constructor===Ob&&a.f===Pb?a=a.b:(xa(a),a="type_error:SafeHtml"),a=encodeURIComponent(String(qd(a)))),/^[\s\xa0]*$/.test(a)||
(a=Yb("IFRAME",{src:'javascript:"<body><img src=\\""+'+a+'+"\\"></body>"',style:"display:none"}),(9==a.nodeType?a:a.ownerDocument||a.document).body.appendChild(a))):e?Ke(a,b,"POST",e,d):Q("USE_NET_AJAX_FOR_PING_TRANSPORT",!1)||d?Ke(a,b,"GET","",d):((d=ie.EXPERIMENT_FLAGS)&&d.web_use_beacon_api_for_ad_click_server_pings&&-1!=I(J(5,a)).indexOf("/aclk")&&"1"===rc(a,"ae")&&"1"===rc(a,"act")?Te(a)?(b&&b(),d=!0):d=!1:d=!1,d||Ue(a,b)))}
function Te(a,b){try{if(window.navigator&&window.navigator.sendBeacon&&window.navigator.sendBeacon(a,void 0===b?"":b))return!0}catch(c){}return!1}
function Ue(a,b){var c=new Image,d=""+Re++;Qe[d]=c;c.onload=c.onerror=function(){b&&Qe[d]&&b();delete Qe[d]};
c.src=a}
;var Ve={},We=0;
function Xe(a,b,c,d,e,f){f=f||{};f.name=c||Q("INNERTUBE_CONTEXT_CLIENT_NAME",1);f.version=d||Q("INNERTUBE_CONTEXT_CLIENT_VERSION",void 0);b=void 0===b?"ERROR":b;e=void 0===e?!1:e;b=void 0===b?"ERROR":b;e=window&&window.yterr||(void 0===e?!1:e)||!1;if(!(!a||!e||5<=We||(e=a.stacktrace,c=a.columnNumber,a.hasOwnProperty("params")&&(d=String(JSON.stringify(a.params)),f.params=d.substr(0,500)),a=Cb(a),e=e||a.stack,d=a.lineNumber.toString(),isNaN(d)||isNaN(c)||(d=d+":"+c),window.yterr&&za(window.yterr)&&window.yterr(a),
Ve[a.message]||0<=e.indexOf("/YouTubeCenter.js")||0<=e.indexOf("/mytube.js")))){b={Va:{a:"logerror",t:"jserror",type:a.name,msg:a.message.substr(0,250),line:d,level:b,"client.name":f.name},B:{url:Q("PAGE_NAME",window.location.href),file:a.fileName},method:"POST"};f.version&&(b["client.version"]=f.version);e&&(b.B.stack=e);for(var g in f)b.B["client."+g]=f[g];if(g=Q("LATEST_ECATCHER_SERVICE_TRACKING_PARAMS",void 0))for(var k in g)b.B[k]=g[k];Je(Q("ECATCHER_REPORT_HOST","")+"/error_204",b);Ve[a.message]=
!0;We++}}
;var Ye=window.yt&&window.yt.msgs_||window.ytcfg&&window.ytcfg.msgs||{};v("yt.msgs_",Ye,void 0);function Ze(a){he(Ye,arguments)}
;function $e(a){a&&(a.dataset?a.dataset[af("loaded")]="true":a.setAttribute("data-loaded","true"))}
function bf(a,b){return a?a.dataset?a.dataset[af(b)]:a.getAttribute("data-"+b):null}
var cf={};function af(a){return cf[a]||(cf[a]=String(a).replace(/\-([a-z])/g,function(a,c){return c.toUpperCase()}))}
;var df=w("ytPubsubPubsubInstance")||new N;N.prototype.subscribe=N.prototype.subscribe;N.prototype.unsubscribeByKey=N.prototype.K;N.prototype.publish=N.prototype.J;N.prototype.clear=N.prototype.clear;v("ytPubsubPubsubInstance",df,void 0);var ef=w("ytPubsubPubsubSubscribedKeys")||{};v("ytPubsubPubsubSubscribedKeys",ef,void 0);var ff=w("ytPubsubPubsubTopicToKeys")||{};v("ytPubsubPubsubTopicToKeys",ff,void 0);var gf=w("ytPubsubPubsubIsSynchronous")||{};v("ytPubsubPubsubIsSynchronous",gf,void 0);
function hf(a,b){var c=jf();if(c){var d=c.subscribe(a,function(){var c=arguments;var f=function(){ef[d]&&b.apply(window,c)};
try{gf[a]?f():T(f,0)}catch(g){R(g)}},void 0);
ef[d]=!0;ff[a]||(ff[a]=[]);ff[a].push(d);return d}return 0}
function kf(a){var b=jf();b&&("number"==typeof a?a=[a]:t(a)&&(a=[parseInt(a,10)]),F(a,function(a){b.unsubscribeByKey(a);delete ef[a]}))}
function lf(a,b){var c=jf();c&&c.publish.apply(c,arguments)}
function mf(a){var b=jf();if(b)if(b.clear(a),a)nf(a);else for(var c in ff)nf(c)}
function jf(){return w("ytPubsubPubsubInstance")}
function nf(a){ff[a]&&(a=ff[a],F(a,function(a){ef[a]&&delete ef[a]}),a.length=0)}
;var of=/\.vflset|-vfl[a-zA-Z0-9_+=-]+/,pf=/-[a-zA-Z]{2,3}_[a-zA-Z]{2,3}(?=(\/|$))/;function qf(a,b,c){c=void 0===c?null:c;if(window.spf){c="";if(a){var d=a.indexOf("jsbin/"),e=a.lastIndexOf(".js"),f=d+6;-1<d&&-1<e&&e>f&&(c=a.substring(f,e),c=c.replace(of,""),c=c.replace(pf,""),c=c.replace("debug-",""),c=c.replace("tracing-",""))}spf.script.load(a,c,b)}else rf(a,b,c)}
function rf(a,b,c){c=void 0===c?null:c;var d=sf(a),e=document.getElementById(d),f=e&&bf(e,"loaded"),g=e&&!f;f?b&&b():(b&&(f=hf(d,b),b=""+(b[Aa]||(b[Aa]=++Ba)),tf[b]=f),g||(e=uf(a,d,function(){bf(e,"loaded")||($e(e),lf(d),T(Ea(mf,d),0))},c)))}
function uf(a,b,c,d){d=void 0===d?null:d;var e=document.createElement("SCRIPT");e.id=b;e.onload=function(){c&&setTimeout(c,0)};
e.onreadystatechange=function(){switch(e.readyState){case "loaded":case "complete":e.onload()}};
d&&e.setAttribute("nonce",d);Sb(e,bc(a));a=document.getElementsByTagName("head")[0]||document.body;a.insertBefore(e,a.firstChild);return e}
function vf(a){a=sf(a);var b=document.getElementById(a);b&&(mf(a),b.parentNode.removeChild(b))}
function wf(a,b){if(a&&b){var c=""+(b[Aa]||(b[Aa]=++Ba));(c=tf[c])&&kf(c)}}
function sf(a){var b=document.createElement("a");Rb(b,a);a=b.href.replace(/^[a-zA-Z]+:\/\//,"//");return"js-"+Xa(a)}
var tf={};function xf(){}
function yf(a,b){return zf(a,1,b)}
;function Af(){}
n(Af,xf);function zf(a,b,c){isNaN(c)&&(c=void 0);var d=w("yt.scheduler.instance.addJob");return d?d(a,b,c):void 0===c?(a(),NaN):T(a,c||0)}
function Bf(a){if(!isNaN(a)){var b=w("yt.scheduler.instance.cancelJob");b?b(a):U(a)}}
Af.prototype.start=function(){var a=w("yt.scheduler.instance.start");a&&a()};
Af.prototype.pause=function(){var a=w("yt.scheduler.instance.pause");a&&a()};
wa(Af);Af.getInstance();var Cf=[],Df=!1;function Ef(){if("1"!=bb(ke(),"args","privembed")){var a=function(){Df=!0;"google_ad_status"in window?P("DCLKSTAT",1):P("DCLKSTAT",2)};
qf("//static.doubleclick.net/instream/ad_status.js",a);Cf.push(yf(function(){Df||"google_ad_status"in window||(wf("//static.doubleclick.net/instream/ad_status.js",a),Df=!0,P("DCLKSTAT",3))},5E3))}}
function Ff(){return parseInt(Q("DCLKSTAT",0),10)}
;function Gf(){this.c=!1;this.b=null}
Gf.prototype.initialize=function(a,b,c,d,e){var f=this;b?(this.c=!0,qf(b,function(){f.c=!1;if(window.botguard)Hf(f,c,d);else{vf(b);var a=Error("Unable to load Botguard");a.params="from "+b;te(a)}},e)):a&&(eval(a),window.botguard?Hf(this,c,d):te(Error("Unable to load Botguard from JS")))};
function Hf(a,b,c){try{a.b=new botguard.bg(b)}catch(d){te(d)}c&&c(b)}
Gf.prototype.dispose=function(){this.b=null};var If=new Gf,Jf=!1,Kf=0,Lf="";function Mf(a){S("botguard_periodic_refresh")?Kf=O():S("botguard_always_refresh")&&(Lf=a)}
function Nf(a){if(a){if(If.c)return!1;if(S("botguard_periodic_refresh"))return 72E5<O()-Kf;if(S("botguard_always_refresh"))return Lf!=a}else return!1;return!Jf}
function Of(){return!!If.b}
function Pf(a){a=void 0===a?{}:a;a=void 0===a?{}:a;return If.b?If.b.invoke(void 0,void 0,a):null}
;var Qf=0;v("ytDomDomGetNextId",w("ytDomDomGetNextId")||function(){return++Qf},void 0);var Rf={stopImmediatePropagation:1,stopPropagation:1,preventMouseEvent:1,preventManipulation:1,preventDefault:1,layerX:1,layerY:1,screenX:1,screenY:1,scale:1,rotation:1,webkitMovementX:1,webkitMovementY:1};
function Sf(a){this.type="";this.state=this.source=this.data=this.currentTarget=this.relatedTarget=this.target=null;this.charCode=this.keyCode=0;this.metaKey=this.shiftKey=this.ctrlKey=this.altKey=!1;this.clientY=this.clientX=0;this.changedTouches=this.touches=null;if(a=a||window.event){this.event=a;for(var b in a)b in Rf||(this[b]=a[b]);(b=a.target||a.srcElement)&&3==b.nodeType&&(b=b.parentNode);this.target=b;if(b=a.relatedTarget)try{b=b.nodeName?b:null}catch(c){b=null}else"mouseover"==this.type?
b=a.fromElement:"mouseout"==this.type&&(b=a.toElement);this.relatedTarget=b;this.clientX=void 0!=a.clientX?a.clientX:a.pageX;this.clientY=void 0!=a.clientY?a.clientY:a.pageY;this.keyCode=a.keyCode?a.keyCode:a.which;this.charCode=a.charCode||("keypress"==this.type?this.keyCode:0);this.altKey=a.altKey;this.ctrlKey=a.ctrlKey;this.shiftKey=a.shiftKey;this.metaKey=a.metaKey;this.b=a.pageX;this.c=a.pageY}}
function Tf(a){if(document.body&&document.documentElement){var b=document.body.scrollTop+document.documentElement.scrollTop;a.b=a.clientX+(document.body.scrollLeft+document.documentElement.scrollLeft);a.c=a.clientY+b}}
Sf.prototype.preventDefault=function(){this.event&&(this.event.returnValue=!1,this.event.preventDefault&&this.event.preventDefault())};
Sf.prototype.stopPropagation=function(){this.event&&(this.event.cancelBubble=!0,this.event.stopPropagation&&this.event.stopPropagation())};
Sf.prototype.stopImmediatePropagation=function(){this.event&&(this.event.cancelBubble=!0,this.event.stopImmediatePropagation&&this.event.stopImmediatePropagation())};var db=w("ytEventsEventsListeners")||{};v("ytEventsEventsListeners",db,void 0);var Uf=w("ytEventsEventsCounter")||{count:0};v("ytEventsEventsCounter",Uf,void 0);
function Vf(a,b,c,d){d=void 0===d?{}:d;a.addEventListener&&("mouseenter"!=b||"onmouseenter"in document?"mouseleave"!=b||"onmouseenter"in document?"mousewheel"==b&&"MozBoxSizing"in document.documentElement.style&&(b="MozMousePixelScroll"):b="mouseout":b="mouseover");return cb(function(e){var f="boolean"==typeof e[4]&&e[4]==!!d,g=y(e[4])&&y(d)&&gb(e[4],d);return!!e.length&&e[0]==a&&e[1]==b&&e[2]==c&&(f||g)})}
var Wf=Eb(function(){var a=!1;try{var b=Object.defineProperty({},"capture",{get:function(){a=!0}});
window.addEventListener("test",null,b)}catch(c){}return a});
function V(a,b,c,d){d=void 0===d?{}:d;if(!a||!a.addEventListener&&!a.attachEvent)return"";var e=Vf(a,b,c,d);if(e)return e;e=++Uf.count+"";var f=!("mouseenter"!=b&&"mouseleave"!=b||!a.addEventListener||"onmouseenter"in document);var g=f?function(d){d=new Sf(d);if(!ac(d.relatedTarget,function(b){return b==a}))return d.currentTarget=a,d.type=b,c.call(a,d)}:function(b){b=new Sf(b);
b.currentTarget=a;return c.call(a,b)};
g=se(g);a.addEventListener?("mouseenter"==b&&f?b="mouseover":"mouseleave"==b&&f?b="mouseout":"mousewheel"==b&&"MozBoxSizing"in document.documentElement.style&&(b="MozMousePixelScroll"),Wf()||"boolean"==typeof d?a.addEventListener(b,g,d):a.addEventListener(b,g,!!d.capture)):a.attachEvent("on"+b,g);db[e]=[a,b,c,g,d];return e}
function Xf(a){a&&("string"==typeof a&&(a=[a]),F(a,function(a){if(a in db){var b=db[a],d=b[0],e=b[1],f=b[3];b=b[4];d.removeEventListener?Wf()||"boolean"==typeof b?d.removeEventListener(e,f,b):d.removeEventListener(e,f,!!b.capture):d.detachEvent&&d.detachEvent("on"+e,f);delete db[a]}}))}
;function Yf(a){this.u=a;this.b=null;this.h=0;this.l=null;this.i=0;this.f=[];for(a=0;4>a;a++)this.f.push(0);this.g=0;this.C=V(window,"mousemove",z(this.F,this));a=z(this.A,this);za(a)&&(a=se(a));this.G=window.setInterval(a,25)}
C(Yf,L);Yf.prototype.F=function(a){r(a.b)||Tf(a);var b=a.b;r(a.c)||Tf(a);this.b=new Tb(b,a.c)};
Yf.prototype.A=function(){if(this.b){var a=O();if(0!=this.h){var b=this.l,c=this.b,d=b.x-c.x;b=b.y-c.y;d=Math.sqrt(d*d+b*b)/(a-this.h);this.f[this.g]=.5<Math.abs((d-this.i)/this.i)?1:0;for(c=b=0;4>c;c++)b+=this.f[c]||0;3<=b&&this.u();this.i=d}this.h=a;this.l=this.b;this.g=(this.g+1)%4}};
Yf.prototype.j=function(){window.clearInterval(this.G);Xf(this.C)};var Zf={};
function $f(a){var b=void 0===a?{}:a;a=void 0===b.xa?!0:b.xa;b=void 0===b.Ka?!1:b.Ka;if(null==w("_lact",window)){var c=parseInt(Q("LACT"),10);c=isFinite(c)?A()-Math.max(c,0):-1;v("_lact",c,window);v("_fact",c,window);-1==c&&ag();V(document,"keydown",ag);V(document,"keyup",ag);V(document,"mousedown",ag);V(document,"mouseup",ag);a&&(b?V(window,"touchmove",function(){bg("touchmove",200)},{passive:!0}):(V(window,"resize",function(){bg("resize",200)}),V(window,"scroll",function(){bg("scroll",200)})));
new Yf(function(){bg("mouse",100)});
V(document,"touchstart",ag,{passive:!0});V(document,"touchend",ag,{passive:!0})}}
function bg(a,b){Zf[a]||(Zf[a]=!0,yf(function(){ag();Zf[a]=!1},b))}
function ag(){null==w("_lact",window)&&$f();var a=A();v("_lact",a,window);-1==w("_fact",window)&&v("_fact",a,window);(a=w("ytglobal.ytUtilActivityCallback_"))&&a()}
function cg(){var a=w("_lact",window);return null==a?-1:Math.max(A()-a,0)}
;var dg=Math.pow(2,16)-1,eg=null,fg=0,gg={log_event:"events",log_interaction:"interactions"},hg=Object.create(null);hg.log_event="GENERIC_EVENT_LOGGING";hg.log_interaction="INTERACTION_LOGGING";var ig=new Set(["log_event"]),jg={},kg=0,lg=0,W=w("ytLoggingTransportLogPayloadsQueue_")||{};v("ytLoggingTransportLogPayloadsQueue_",W,void 0);var mg=w("ytLoggingTransportTokensToCttTargetIds_")||{};v("ytLoggingTransportTokensToCttTargetIds_",mg,void 0);var ng=w("ytLoggingTransportDispatchedStats_")||{};
v("ytLoggingTransportDispatchedStats_",ng,void 0);v("ytytLoggingTransportCapturedTime_",w("ytLoggingTransportCapturedTime_")||{},void 0);function og(){U(kg);U(lg);lg=0;if(!eb(W)){for(var a in W){var b=jg[a];b&&(pg(a,b),delete W[a])}eb(W)||qg()}}
function qg(){S("web_gel_timeout_cap")&&!lg&&(lg=T(og,3E4));U(kg);kg=T(og,Q("LOGGING_BATCH_TIMEOUT",1E4))}
function rg(a,b){b=void 0===b?"":b;W[a]=W[a]||{};W[a][b]=W[a][b]||[];return W[a][b]}
function pg(a,b){var c=gg[a],d=ng[a]||{};ng[a]=d;var e=Math.round(O());for(m in W[a]){var f=b.b;f={client:{hl:f.Da,gl:f.Ca,clientName:f.Aa,clientVersion:f.Ba}};var g=window.devicePixelRatio;g&&1!=g&&(f.client.screenDensityFloat=String(g));Q("DELEGATED_SESSION_ID")&&!S("pageid_as_header_web")&&(f.user={onBehalfOfUser:Q("DELEGATED_SESSION_ID")});f={context:f};f[c]=rg(a,m);d.dispatchedEventCount=d.dispatchedEventCount||0;d.dispatchedEventCount+=f[c].length;if(g=mg[m])a:{var k=m;if(g.videoId)var l="VIDEO";
else if(g.playlistId)l="PLAYLIST";else break a;f.credentialTransferTokenTargetId=g;f.context=f.context||{};f.context.user=f.context.user||{};f.context.user.credentialTransferTokens=[{token:k,scope:l}]}delete mg[m];f.requestTimeMs=e;if(g=je("EVENT_ID"))l=(Q("BATCH_CLIENT_COUNTER",void 0)||0)+1,l>dg&&(l=1),P("BATCH_CLIENT_COUNTER",l),g={serializedEventId:g,clientCounter:l},f.serializedClientEventId=g,eg&&fg&&S("log_gel_rtt_web")&&(f.previousBatchInfo={serializedClientEventId:eg,roundtripMs:fg}),eg=
g,fg=0;sg(b,a,f,{retry:ig.has(a),onSuccess:z(tg,this,O())})}if(d.previousDispatchMs){c=e-d.previousDispatchMs;var m=d.diffCount||0;d.averageTimeBetweenDispatchesMs=m?(d.averageTimeBetweenDispatchesMs*m+c)/(m+1):c;d.diffCount=m+1}d.previousDispatchMs=e}
function tg(a){fg=Math.round(O()-a)}
;function ug(a,b,c,d,e){var f={};f.eventTimeMs=Math.round(d||O());f[a]=b;f.context={lastActivityMs:String(d?-1:cg())};e?(a={},e.videoId?a.videoId=e.videoId:e.playlistId&&(a.playlistId=e.playlistId),mg[e.token]=a,e=rg("log_event",e.token)):e=rg("log_event");e.push(f);c&&(jg.log_event=new c);e.length>=(Number(S("web_logging_max_batch")||0)||20)?og():qg()}
;function vg(a,b,c){c=void 0===c?{}:c;var d={"X-Goog-Visitor-Id":c.visitorData||Q("VISITOR_DATA","")};if(b&&b.includes("www.youtube-nocookie.com"))return d;(b=c.jb||Q("AUTHORIZATION"))||(a?b="Bearer "+w("gapi.auth.getToken")().ib:b=Kc([]));b&&(d.Authorization=b,d["X-Goog-AuthUser"]=Q("SESSION_INDEX",0),S("pageid_as_header_web")&&(d["X-Goog-PageId"]=Q("DELEGATED_SESSION_ID")));return d}
function wg(a){a=Object.assign({},a);delete a.Authorization;var b=Kc();if(b){var c=new $c;c.update(Q("INNERTUBE_API_KEY",void 0));c.update(b);b=c.digest();ya(b);if(!yb)for(yb={},zb={},c=0;65>c;c++)yb[c]="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=".charAt(c),zb[c]="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_.".charAt(c);c=zb;for(var d=[],e=0;e<b.length;e+=3){var f=b[e],g=e+1<b.length,k=g?b[e+1]:0,l=e+2<b.length,m=l?b[e+2]:0,u=f>>2;f=(f&3)<<4|k>>4;k=(k&15)<<
2|m>>6;m&=63;l||(m=64,g||(k=64));d.push(c[u],c[f],c[k],c[m])}a.hash=d.join("")}return a}
;function xg(a,b,c,d){Bb.set(""+a,b,c,"/",void 0===d?"youtube.com":d,!1)}
;function yg(){var a=new Zd;(a=a.isAvailable()?new ee(a,"yt.innertube"):null)||(a=new $d("yt.innertube"),a=a.isAvailable()?a:null);this.b=a?new Vd(a):null;this.c=document.domain||window.location.hostname}
yg.prototype.set=function(a,b,c,d){c=c||31104E3;this.remove(a);if(this.b)try{this.b.set(a,b,A()+1E3*c);return}catch(f){}var e="";if(d)try{e=escape(qd(b))}catch(f){return}else e=escape(b);xg(a,e,c,this.c)};
yg.prototype.get=function(a,b){var c=void 0,d=!this.b;if(!d)try{c=this.b.get(a)}catch(e){d=!0}if(d&&(c=Bb.get(""+a,void 0))&&(c=unescape(c),b))try{c=JSON.parse(c)}catch(e){this.remove(a),c=void 0}return c};
yg.prototype.remove=function(a){this.b&&this.b.remove(a);var b=this.c;Bb.remove(""+a,"/",void 0===b?"youtube.com":b)};var zg=new yg;function Ag(a,b,c,d){if(d)return null;d=zg.get("nextId",!0)||1;var e=zg.get("requests",!0)||{};e[d]={method:a,request:b,authState:wg(c),requestTime:Math.round(O())};zg.set("nextId",d+1,86400,!0);zg.set("requests",e,86400,!0);return d}
function Bg(a){var b=zg.get("requests",!0)||{};delete b[a];zg.set("requests",b,86400,!0)}
function Cg(a){var b=zg.get("requests",!0);if(b){for(var c in b){var d=b[c];if(!(6E4>Math.round(O())-d.requestTime)){var e=d.authState,f=wg(vg(!1));gb(e,f)&&(e=d.request,"requestTimeMs"in e&&(e.requestTimeMs=Math.round(O())),sg(a,d.method,e,{}));delete b[c]}}zg.set("requests",b,86400,!0)}}
;function Dg(a){var b=this;this.b=a||{ya:je("INNERTUBE_API_KEY"),za:je("INNERTUBE_API_VERSION"),Aa:Q("INNERTUBE_CONTEXT_CLIENT_NAME","WEB"),Ba:je("INNERTUBE_CONTEXT_CLIENT_VERSION"),Da:je("INNERTUBE_CONTEXT_HL"),Ca:je("INNERTUBE_CONTEXT_GL"),Ea:je("INNERTUBE_HOST_OVERRIDE")||"",Fa:!!Q("INNERTUBE_USE_THIRD_PARTY_AUTH",!1)};zf(function(){Cg(b)},0,5E3)}
function sg(a,b,c,d){!Q("VISITOR_DATA")&&.01>Math.random()&&R(Error("Missing VISITOR_DATA when sending innertube request."),"WARNING");var e={headers:{"Content-Type":"application/json"},method:"POST",B:c,ia:"JSON",L:function(){d.L()},
ha:d.L,onSuccess:function(a,b){if(d.onSuccess)d.onSuccess(b)},
ga:function(a){if(d.onSuccess)d.onSuccess(a)},
onError:function(a,b){if(d.onError)d.onError(b)},
ob:function(a){if(d.onError)d.onError(a)},
timeout:d.timeout,withCredentials:!0},f="",g=a.b.Ea;g&&(f=g);g=a.b.Fa||!1;var k=vg(g,f,d);Object.assign(e.headers,k);e.headers.Authorization&&!f&&(e.headers["x-origin"]=window.location.origin);var l=""+f+("/youtubei/"+a.b.za+"/"+b)+"?alt=json&key="+a.b.ya,m;if(d.retry&&S("retry_web_logging_batches")&&"www.youtube-nocookie.com"!=f&&(m=Ag(b,c,k,g))){var u=e.onSuccess,da=e.ga;e.onSuccess=function(a,b){Bg(m);u(a,b)};
c.ga=function(a,b){Bg(m);da(a,b)}}try{S("use_fetch_for_op_xhr")?Ge(l,e):Pe(l,e)}catch(Dd){if("InvalidAccessError"==Dd)m&&(Bg(m),m=0),R(Error("An extension is blocking network request."),"WARNING");
else throw Dd;}m&&zf(function(){Cg(a)},0,5E3)}
;var Eg=A().toString();
function Fg(){a:{if(window.crypto&&window.crypto.getRandomValues)try{var a=Array(16),b=new Uint8Array(16);window.crypto.getRandomValues(b);for(var c=0;c<a.length;c++)a[c]=b[c];var d=a;break a}catch(e){}d=Array(16);for(a=0;16>a;a++){b=A();for(c=0;c<b%23;c++)d[a]=Math.random();d[a]=Math.floor(256*Math.random())}if(Eg)for(a=1,b=0;b<Eg.length;b++)d[a%16]=d[a%16]^d[(a-1)%16]/4^Eg.charCodeAt(b),a++}a=[];for(b=0;b<d.length;b++)a.push("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_".charAt(d[b]&63));
return a.join("")}
;var Gg=w("ytLoggingTimeDocumentNonce_")||Fg();v("ytLoggingTimeDocumentNonce_",Gg,void 0);function Hg(a){this.b=a}
function Ig(a){var b={};void 0!==a.b.trackingParams?b.trackingParams=a.b.trackingParams:(b.veType=a.b.veType,null!=a.b.veCounter&&(b.veCounter=a.b.veCounter),null!=a.b.elementIndex&&(b.elementIndex=a.b.elementIndex));void 0!==a.b.dataElement&&(b.dataElement=Ig(a.b.dataElement));void 0!==a.b.youtubeData&&(b.youtubeData=a.b.youtubeData);return b}
Hg.prototype.toString=function(){return JSON.stringify(Ig(this))};
var Jg=1;function Kg(a){a=void 0===a?0:a;return 0==a?"client-screen-nonce":"client-screen-nonce."+a}
function Lg(a){a=void 0===a?0:a;return 0==a?"ROOT_VE_TYPE":"ROOT_VE_TYPE."+a}
function Mg(a){return Q(Lg(void 0===a?0:a),void 0)}
v("yt_logging_screen.getRootVeType",Mg,void 0);function Ng(){var a=Mg(0),b;a?b=new Hg({veType:a,youtubeData:void 0}):b=null;return b}
function Og(a){a=void 0===a?0:a;var b=Q(Kg(a));b||0!=a||(b=Q("EVENT_ID"));return b?b:null}
v("yt_logging_screen.getCurrentCsn",Og,void 0);function Pg(a,b,c){c=void 0===c?0:c;if(a!==Q(Kg(c))||b!==Q(Lg(c)))P(Kg(c),a),P(Lg(c),b),0==c&&(b=function(){setTimeout(function(){a&&ug("foregroundHeartbeatScreenAssociated",{clientDocumentNonce:Gg,clientScreenNonce:a},Dg)},0)},"requestAnimationFrame"in window?window.requestAnimationFrame(b):b())}
v("yt_logging_screen.setCurrentScreen",Pg,void 0);function Qg(a,b,c){b=void 0===b?{}:b;c=void 0===c?!1:c;var d=Q("EVENT_ID");d&&(b.ei||(b.ei=d));if(b){d=a;var e=Q("VALID_SESSION_TEMPDATA_DOMAINS",[]),f=I(J(3,window.location.href));f&&e.push(f);f=I(J(3,d));if(0<=Ga(e,f)||!f&&0==d.lastIndexOf("/",0))if(S("autoescape_tempdata_url")&&(e=document.createElement("a"),Rb(e,d),d=e.href),d){f=d.match(mc);d=f[5];e=f[6];f=f[7];var g="";d&&(g+=d);e&&(g+="?"+e);f&&(g+="#"+f);d=g;e=d.indexOf("#");if(d=0>e?d:d.substr(0,e)){if(b.itct||b.ved)b.csn=b.csn||Og();if(k){var k=
parseInt(k,10);isFinite(k)&&0<k&&(d="ST-"+Xa(d).toString(36),b=b?oc(b):"",xg(d,b,k||5))}else k="ST-"+Xa(d).toString(36),b=b?oc(b):"",xg(k,b,5)}}}if(c)return!1;if((window.ytspf||{}).enabled)spf.navigate(a);else{var l=void 0===l?{}:l;var m=void 0===m?"":m;var u=void 0===u?window:u;c=u.location;a=pc(a,l)+m;a=a instanceof H?a:Mb(a);c.href=Kb(a)}return!0}
;function Rg(a,b,c){a={csn:a,parentVe:Ig(b),childVes:Ia(c,function(a){return Ig(a)})};
ug("visualElementAttached",a,Dg)}
;function Sg(a){a=a||{};this.url=a.url||"";this.args=a.args||hb(Tg);this.assets=a.assets||{};this.attrs=a.attrs||hb(Ug);this.fallback=a.fallback||null;this.fallbackMessage=a.fallbackMessage||null;this.html5=!!a.html5;this.disable=a.disable||{};this.loaded=!!a.loaded;this.messages=a.messages||{}}
var Tg={enablejsapi:1},Ug={};Sg.prototype.clone=function(){var a=new Sg,b;for(b in this)if(this.hasOwnProperty(b)){var c=this[b];"object"==xa(c)?a[b]=hb(c):a[b]=c}return a};function Vg(){L.call(this);this.b=[]}
n(Vg,L);Vg.prototype.j=function(){for(;this.b.length;){var a=this.b.pop();a.target.removeEventListener(a.name,a.lb)}L.prototype.j.call(this)};var Wg=/cssbin\/(?:debug-)?([a-zA-Z0-9_-]+?)(?:-2x|-web|-rtl|-vfl|.css)/;function Xg(a){a=a||"";if(window.spf){var b=a.match(Wg);spf.style.load(a,b?b[1]:"",void 0)}else Yg(a)}
function Yg(a){var b=Zg(a),c=document.getElementById(b),d=c&&bf(c,"loaded");d||c&&!d||(c=$g(a,b,function(){bf(c,"loaded")||($e(c),lf(b),T(Ea(mf,b),0))}))}
function $g(a,b,c){var d=document.createElement("link");d.id=b;d.onload=function(){c&&setTimeout(c,0)};
a=bc(a);d.rel="stylesheet";d.href=Ib(a);(document.getElementsByTagName("head")[0]||document.body).appendChild(d);return d}
function Zg(a){var b=document.createElement("A");a=Nb(a);Rb(b,a);b=b.href.replace(/^[a-zA-Z]+:\/\//,"//");return"css-"+Xa(b)}
;function ah(a,b){L.call(this);this.i=this.R=a;this.C=b;this.l=!1;this.api={};this.O=this.A=null;this.F=new N;bd(this,Ea(cd,this.F));this.g={};this.M=this.P=this.f=this.W=this.b=null;this.G=!1;this.h=this.u=null;this.T={};this.ma=["onReady"];this.V=null;this.ea=NaN;this.N={};bh(this);this.U("WATCH_LATER_VIDEO_ADDED",this.Ha.bind(this));this.U("WATCH_LATER_VIDEO_REMOVED",this.Ia.bind(this));this.U("onAdAnnounce",this.qa.bind(this));this.na=new Vg(this);bd(this,Ea(cd,this.na))}
n(ah,L);h=ah.prototype;
h.ca=function(a){if(!this.c){a instanceof Sg||(a=new Sg(a));this.W=a;this.b=a.clone();this.f=this.b.attrs.id||this.f;"video-player"==this.f&&(this.f=this.C,this.b.attrs.id=this.C);this.i.id==this.f&&(this.f+="-player",this.b.attrs.id=this.f);this.b.args.enablejsapi="1";this.b.args.playerapiid=this.C;this.P||(this.P=ch(this,this.b.args.jsapicallback||"onYouTubePlayerReady"));this.b.args.jsapicallback=null;if(a=this.b.attrs.width)this.i.style.width=kc(Number(a)||a);if(a=this.b.attrs.height)this.i.style.height=kc(Number(a)||
a);dh(this);this.l&&eh(this)}};
h.ta=function(){return this.W};
function eh(a){a.b.loaded||(a.b.loaded=!0,"0"!=a.b.args.autoplay?a.api.loadVideoByPlayerVars(a.b.args):a.api.cueVideoByPlayerVars(a.b.args))}
function fh(a){var b=!0,c=gh(a);c&&a.b&&(a=a.b,b=bf(c,"version")==a.assets.js);return b&&!!w("yt.player.Application.create")}
function dh(a){if(!a.c&&!a.G){var b=fh(a);if(b&&"html5"==(gh(a)?"html5":null))a.M="html5",a.l||hh(a);else if(ih(a),a.M="html5",b&&a.h)a.R.appendChild(a.h),hh(a);else{a.b.loaded=!0;var c=!1;a.u=function(){c=!0;var b=a.b.clone();w("yt.player.Application.create")(a.R,b);hh(a)};
a.G=!0;b?a.u():(qf(a.b.assets.js,a.u),Xg(a.b.assets.css),jh(a)&&!c&&v("yt.player.Application.create",null,void 0))}}}
function gh(a){var b=Vb(a.f);!b&&a.i&&a.i.querySelector&&(b=a.i.querySelector("#"+a.f));return b}
function hh(a){if(!a.c){var b=gh(a),c=!1;b&&b.getApiInterface&&b.getApiInterface()&&(c=!0);c?(a.G=!1,b.isNotServable&&b.isNotServable(a.b.args.video_id)||kh(a)):a.ea=T(function(){hh(a)},50)}}
function kh(a){bh(a);a.l=!0;var b=gh(a);b.addEventListener&&(a.A=lh(a,b,"addEventListener"));b.removeEventListener&&(a.O=lh(a,b,"removeEventListener"));var c=b.getApiInterface();c=c.concat(b.getInternalApiInterface());for(var d=0;d<c.length;d++){var e=c[d];a.api[e]||(a.api[e]=lh(a,b,e))}for(var f in a.g)a.A(f,a.g[f]);eh(a);a.P&&a.P(a.api);a.F.J("onReady",a.api)}
function lh(a,b,c){var d=b[c];return function(){try{return a.V=null,d.apply(b,arguments)}catch(e){"sendAbandonmentPing"!=c&&(e.message+=" ("+c+")",a.V=e,te(e))}}}
function bh(a){a.l=!1;if(a.O)for(var b in a.g)a.O(b,a.g[b]);for(var c in a.N)U(parseInt(c,10));a.N={};a.A=null;a.O=null;for(var d in a.api)a.api[d]=null;a.api.addEventListener=a.U.bind(a);a.api.removeEventListener=a.Ma.bind(a);a.api.destroy=a.dispose.bind(a);a.api.getLastError=a.ua.bind(a);a.api.getPlayerType=a.va.bind(a);a.api.getCurrentVideoConfig=a.ta.bind(a);a.api.loadNewVideoConfig=a.ca.bind(a);a.api.isReady=a.Ga.bind(a)}
h.Ga=function(){return this.l};
h.U=function(a,b){var c=this,d=ch(this,b);if(d){if(!(0<=Ga(this.ma,a)||this.g[a])){var e=mh(this,a);this.A&&this.A(a,e)}this.F.subscribe(a,d);"onReady"==a&&this.l&&T(function(){d(c.api)},0)}};
h.Ma=function(a,b){if(!this.c){var c=ch(this,b);c&&Pd(this.F,a,c)}};
function ch(a,b){var c=b;if("string"==typeof b){if(a.T[b])return a.T[b];c=function(){var a=w(b);a&&a.apply(q,arguments)};
a.T[b]=c}return c?c:null}
function mh(a,b){var c="ytPlayer"+b+a.C;a.g[b]=c;q[c]=function(c){var d=T(function(){if(!a.c){a.F.J(b,c);var e=a.N,g=String(d);g in e&&delete e[g]}},0);
fb(a.N,String(d))};
return c}
h.qa=function(a){lf("a11y-announce",a)};
h.Ha=function(a){lf("WATCH_LATER_VIDEO_ADDED",a)};
h.Ia=function(a){lf("WATCH_LATER_VIDEO_REMOVED",a)};
h.va=function(){return this.M||(gh(this)?"html5":null)};
h.ua=function(){return this.V};
function ih(a){a.cancel();bh(a);a.M=null;a.b&&(a.b.loaded=!1);var b=gh(a);b&&(fh(a)||!jh(a)?a.h=b:(b&&b.destroy&&b.destroy(),a.h=null));for(a=a.R;b=a.firstChild;)a.removeChild(b)}
h.cancel=function(){this.u&&wf(this.b.assets.js,this.u);U(this.ea);this.G=!1};
h.j=function(){ih(this);if(this.h&&this.b&&this.h.destroy)try{this.h.destroy()}catch(b){R(b)}this.T=null;for(var a in this.g)q[this.g[a]]=null;this.W=this.b=this.api=null;delete this.R;delete this.i;L.prototype.j.call(this)};
function jh(a){return a.b&&a.b.args&&a.b.args.fflags?-1!=a.b.args.fflags.indexOf("player_destroy_old_version=true"):!1}
;var nh={},oh="player_uid_"+(1E9*Math.random()>>>0);function ph(a){var b="player";b=t(b)?Vb(b):b;var c=oh+"_"+(b[Aa]||(b[Aa]=++Ba)),d=nh[c];if(d)return d.ca(a),d.api;d=new ah(b,c);nh[c]=d;lf("player-added",d.api);bd(d,Ea(qh,d));T(function(){d.ca(a)},0);
return d.api}
function qh(a){delete nh[a.C]}
;function rh(a,b,c){var d=Dg;Q("ytLoggingEventsDefaultDisabled",!1)&&Dg==Dg&&(d=null);ug(a,b,d,c,void 0)}
;var sh=w("ytLoggingLatencyUsageStats_")||{};v("ytLoggingLatencyUsageStats_",sh,void 0);var th=0;function uh(a){sh[a]=sh[a]||{count:0};var b=sh[a];b.count++;b.time=O();th||(th=zf(vh,0,5E3));return 10<b.count?(11==b.count&&Xe(Error("CSI data exceeded logging limit with key: "+a),0==a.indexOf("info")?"WARNING":"ERROR"),!0):!1}
function vh(){var a=O(),b;for(b in sh)6E4<a-sh[b].time&&delete sh[b];th=0}
;function wh(a,b){this.version=a;this.args=b}
;function xh(a){this.topic=a}
xh.prototype.toString=function(){return this.topic};var yh=w("ytPubsub2Pubsub2Instance")||new N;N.prototype.subscribe=N.prototype.subscribe;N.prototype.unsubscribeByKey=N.prototype.K;N.prototype.publish=N.prototype.J;N.prototype.clear=N.prototype.clear;v("ytPubsub2Pubsub2Instance",yh,void 0);v("ytPubsub2Pubsub2SubscribedKeys",w("ytPubsub2Pubsub2SubscribedKeys")||{},void 0);v("ytPubsub2Pubsub2TopicToKeys",w("ytPubsub2Pubsub2TopicToKeys")||{},void 0);v("ytPubsub2Pubsub2IsAsync",w("ytPubsub2Pubsub2IsAsync")||{},void 0);
v("ytPubsub2Pubsub2SkipSubKey",null,void 0);function zh(a,b){var c=w("ytPubsub2Pubsub2Instance");c&&c.publish.call(c,a.toString(),a,b)}
;var X=window.performance||window.mozPerformance||window.msPerformance||window.webkitPerformance||{};function Ah(){var a=Q("TIMING_TICK_EXPIRATION");a||(a={},P("TIMING_TICK_EXPIRATION",a));return a}
function Bh(){var a=Ah(),b;for(b in a)Bf(a[b]);P("TIMING_TICK_EXPIRATION",{})}
;function Ch(a,b){wh.call(this,1,arguments)}
n(Ch,wh);function Dh(a,b){wh.call(this,1,arguments)}
n(Dh,wh);var Eh=new xh("aft-recorded"),Fh=new xh("timing-sent");var Gh={vc:!0},Y={},Hh=(Y.ad_allowed="adTypesAllowed",Y.yt_abt="adBreakType",Y.ad_cpn="adClientPlaybackNonce",Y.ad_docid="adVideoId",Y.yt_ad_an="adNetworks",Y.ad_at="adType",Y.browse_id="browseId",Y.p="httpProtocol",Y.t="transportProtocol",Y.cpn="clientPlaybackNonce",Y.csn="clientScreenNonce",Y.docid="videoId",Y.is_continuation="isContinuation",Y.is_nav="isNavigation",Y.b_p="kabukiInfo.browseParams",Y.is_prefetch="kabukiInfo.isPrefetch",Y.is_secondary_nav="kabukiInfo.isSecondaryNav",Y.prev_browse_id=
"kabukiInfo.prevBrowseId",Y.query_source="kabukiInfo.querySource",Y.voz_type="kabukiInfo.vozType",Y.yt_lt="loadType",Y.yt_ad="isMonetized",Y.nr="webInfo.navigationReason",Y.ncnp="webInfo.nonPreloadedNodeCount",Y.paused="playerInfo.isPausedOnLoad",Y.yt_pt="playerType",Y.fmt="playerInfo.itag",Y.yt_pl="watchInfo.isPlaylist",Y.yt_ad_pr="prerollAllowed",Y.pa="previousAction",Y.yt_red="isRedSubscriber",Y.st="serverTimeMs",Y.aq="tvInfo.appQuality",Y.br_trs="tvInfo.bedrockTriggerState",Y.label="tvInfo.label",
Y.is_mdx="tvInfo.isMdx",Y.preloaded="tvInfo.isPreloaded",Y.query="unpluggedInfo.query",Y.upg_chip_ids_string="unpluggedInfo.upgChipIdsString",Y.yt_vst="videoStreamType",Y.vph="viewportHeight",Y.vpw="viewportWidth",Y.yt_vis="isVisible",Y),Ih="ap c cver cbrand cmodel ei srt yt_fss yt_li plid vpil vpni vpst yt_eil vpni2 vpil2 icrc icrt pa GetBrowse_rid GetPlayer_rid GetSearch_rid GetWatchNext_rid cmt d_vpct d_vpnfi d_vpni pc pfa pfeh pftr prerender psc rc start tcrt tcrc ssr vpr vps yt_abt yt_fn yt_fs yt_pft yt_pre yt_pt yt_pvis yt_ref yt_sts".split(" "),
Jh="isContinuation isNavigation kabukiInfo.isPrefetch kabukiInfo.isSecondaryNav isMonetized playerInfo.isPausedOnLoad prerollAllowed isRedSubscriber tvInfo.isMdx tvInfo.isPreloaded isVisible watchInfo.isPlaylist".split(" "),Kh={},Lh=(Kh.pa="LATENCY_ACTION_",Kh.yt_pt="LATENCY_PLAYER_",Kh.yt_vst="VIDEO_STREAM_TYPE_",Kh),Mh=!1;
function Nh(){var a=Oh().info.yt_lt="hot_bg";Ph().info_yt_lt=a;if(Qh())if("yt_lt"in Hh){var b=Hh.yt_lt;0<=Ga(Jh,b)&&(a=!!a);"yt_lt"in Lh&&(a=Lh.yt_lt+a.toUpperCase());var c=a;if(Qh()){a={};b=b.split(".");for(var d=a,e=0;e<b.length-1;e++)d[b[e]]=d[b[e]]||{},d=d[b[e]];d[b[b.length-1]]=c;c=Rh();b=Object.keys(a).join("");uh("info_"+b+"_"+c)||(a.clientActionNonce=c,rh("latencyActionInfo",a))}}else 0<=Ga(Ih,"yt_lt")||R(Error("Unknown label yt_lt logged with GEL CSI."))}
function Sh(){var a=Th();if(a.aft)return a.aft;for(var b=Q("TIMING_AFT_KEYS",["ol"]),c=b.length,d=0;d<c;d++){var e=a[b[d]];if(e)return e}return NaN}
z(X.clearResourceTimings||X.webkitClearResourceTimings||X.mozClearResourceTimings||X.msClearResourceTimings||X.oClearResourceTimings||va,X);function Rh(){var a=Oh().nonce;a||(a=Fg(),Oh().nonce=a);return a}
function Th(){return Oh().tick}
function Ph(){var a=Oh();"gel"in a||(a.gel={});return a.gel}
function Oh(){var a;(a=w("ytcsi.data_"))||(a={tick:{},info:{}},B("ytcsi.data_",a));return a}
function Uh(){var a=Th(),b=a.pbr,c=a.vc;a=a.pbs;return b&&c&&a&&b<c&&c<a&&1==Oh().info.yt_pvis}
function Qh(){return!!S("csi_on_gel")||!!Oh().useGel}
function Vh(){Bh();if(!Qh()){var a=Th(),b=Oh().info,c=a._start;for(f in a)if(0==f.lastIndexOf("_",0)&&x(a[f])){var d=f.slice(1);if(d in Gh){var e=Ia(a[f],function(a){return Math.round(a-c)});
b["all_"+d]=e.join()}delete a[f]}e=Q("CSI_SERVICE_NAME","youtube");var f={v:2,s:e,action:Q("TIMING_ACTION",void 0)};d=Nh.srt;void 0!==a.srt&&delete b.srt;if(b.h5jse){var g=window.location.protocol+w("ytplayer.config.assets.js");(g=X.getEntriesByName?X.getEntriesByName(g)[0]:null)?b.h5jse=Math.round(b.h5jse-g.responseEnd):delete b.h5jse}a.aft=Sh();Uh()&&"youtube"==e&&(Nh(),e=a.vc,g=a.pbs,delete a.aft,b.aft=Math.round(g-e));for(var k in b)"_"!=k.charAt(0)&&(f[k]=b[k]);a.ps=O();k={};e=[];for(var l in a)"_"!=
l.charAt(0)&&(g=Math.round(a[l]-c),k[l]=g,e.push(l+"."+g));f.rt=e.join(",");(a=w("ytdebug.logTiming"))&&a(f,k);Wh(f,!!b.ap);zh(Fh,new Dh(k.aft+(d||0),void 0))}}
function Wh(a,b){if(S("debug_csi_data")){var c=w("yt.timing.csiData");c||(c=[],v("yt.timing.csiData",c,void 0));c.push({page:location.href,time:new Date,args:a})}c="";for(var d in a)c+="&"+d+"="+a[d];d="/csi_204?"+c.substring(1);if(window.navigator&&window.navigator.sendBeacon&&b){var e=void 0===e?"":e;Te(d,e)||Se(d,void 0,void 0,void 0,e)}else Se(d);B("yt.timing.pingSent_",!0)}
;function Xh(a){return(0==a.search("cue")||0==a.search("load"))&&"loadModule"!=a}
function Yh(a,b,c){t(a)&&(a={mediaContentUrl:a,startSeconds:b,suggestedQuality:c});b=/\/([ve]|embed)\/([^#?]+)/.exec(a.mediaContentUrl);a.videoId=b&&b[2]?b[2]:null;return Zh(a)}
function Zh(a,b,c){if(y(a)){b=["endSeconds","startSeconds","mediaContentUrl","suggestedQuality","videoId"];c={};for(var d=0;d<b.length;d++){var e=b[d];a[e]&&(c[e]=a[e])}return c}return{videoId:a,startSeconds:b,suggestedQuality:c}}
function $h(a,b,c,d){if(y(a)&&!x(a)){b="playlist list listType index startSeconds suggestedQuality".split(" ");c={};for(d=0;d<b.length;d++){var e=b[d];a[e]&&(c[e]=a[e])}return c}b={index:b,startSeconds:c,suggestedQuality:d};t(a)&&16==a.length?b.list="PL"+a:b.playlist=a;return b}
;function ai(a){L.call(this);this.f=a;this.f.subscribe("command",this.ja,this);this.g={};this.h=!1}
C(ai,L);h=ai.prototype;h.start=function(){this.h||this.c||(this.h=!0,bi(this.f,"RECEIVING"))};
h.ja=function(a,b,c){if(this.h&&!this.c){var d=b||{};switch(a){case "addEventListener":t(d.event)&&(a=d.event,a in this.g||(c=z(this.Oa,this,a),this.g[a]=c,this.addEventListener(a,c)));break;case "removeEventListener":t(d.event)&&ci(this,d.event);break;default:if(d=this.b.isReady())d=c||null,d=this.b.shouldUseNewAvailabilityCheck()?this.b.isExternalMethodAvailable(a,d):!!this.b[a];d&&(b=di(a,b||{}),c=this.b.handleExternalCall(a,b,c||null),(c=ei(a,c))&&this.h&&!this.c&&bi(this.f,a,c))}}};
h.Oa=function(a,b){this.h&&!this.c&&bi(this.f,a,this.Y(a,b))};
h.Y=function(a,b){if(null!=b)return{value:b}};
function ci(a,b){b in a.g&&(a.removeEventListener(b,a.g[b]),delete a.g[b])}
h.j=function(){var a=this.f;a.c||Pd(a.b,"command",this.ja,this);this.f=null;for(var b in this.g)ci(this,b);ai.w.j.call(this)};function fi(a,b){ai.call(this,b);this.b=a;this.start()}
C(fi,ai);fi.prototype.addEventListener=function(a,b){this.b.addEventListener(a,b)};
fi.prototype.removeEventListener=function(a,b){this.b.removeEventListener(a,b)};
function di(a,b){switch(a){case "loadVideoById":return b=Zh(b),[b];case "cueVideoById":return b=Zh(b),[b];case "loadVideoByPlayerVars":return[b];case "cueVideoByPlayerVars":return[b];case "loadPlaylist":return b=$h(b),[b];case "cuePlaylist":return b=$h(b),[b];case "seekTo":return[b.seconds,b.allowSeekAhead];case "playVideoAt":return[b.index];case "setVolume":return[b.volume];case "setPlaybackQuality":return[b.suggestedQuality];case "setPlaybackRate":return[b.suggestedRate];case "setLoop":return[b.loopPlaylists];
case "setShuffle":return[b.shufflePlaylist];case "getOptions":return[b.module];case "getOption":return[b.module,b.option];case "setOption":return[b.module,b.option,b.value];case "handleGlobalKeyDown":return[b.keyCode,b.shiftKey,b.ctrlKey,b.altKey,b.metaKey]}return[]}
function ei(a,b){switch(a){case "isMuted":return{muted:b};case "getVolume":return{volume:b};case "getPlaybackRate":return{playbackRate:b};case "getAvailablePlaybackRates":return{availablePlaybackRates:b};case "getVideoLoadedFraction":return{videoLoadedFraction:b};case "getPlayerState":return{playerState:b};case "getCurrentTime":return{currentTime:b};case "getPlaybackQuality":return{playbackQuality:b};case "getAvailableQualityLevels":return{availableQualityLevels:b};case "getDuration":return{duration:b};
case "getVideoUrl":return{videoUrl:b};case "getVideoEmbedCode":return{videoEmbedCode:b};case "getPlaylist":return{playlist:b};case "getPlaylistIndex":return{playlistIndex:b};case "getOptions":return{options:b};case "getOption":return{option:b}}}
fi.prototype.Y=function(a,b){switch(a){case "onReady":return;case "onStateChange":return{playerState:b};case "onPlaybackQualityChange":return{playbackQuality:b};case "onPlaybackRateChange":return{playbackRate:b};case "onError":return{errorCode:b}}return fi.w.Y.call(this,a,b)};
fi.prototype.j=function(){fi.w.j.call(this);delete this.b};function gi(a,b,c,d){L.call(this);this.f=b||null;this.u="*";this.g=c||null;this.sessionId=null;this.channel=d||null;this.C=!!a;this.l=z(this.A,this);window.addEventListener("message",this.l)}
n(gi,L);gi.prototype.A=function(a){if(!("*"!=this.g&&a.origin!=this.g||this.f&&a.source!=this.f)&&t(a.data)){try{var b=JSON.parse(a.data)}catch(c){return}if(!(null==b||this.C&&(this.sessionId&&this.sessionId!=b.id||this.channel&&this.channel!=b.channel))&&b)switch(b.event){case "listening":"null"!=a.origin&&(this.g=this.u=a.origin);this.f=a.source;this.sessionId=b.id;this.b&&(this.b(),this.b=null);break;case "command":this.h&&(!this.i||0<=Ga(this.i,b.func))&&this.h(b.func,b.args,a.origin)}}};
gi.prototype.sendMessage=function(a,b){var c=b||this.f;if(c){this.sessionId&&(a.id=this.sessionId);this.channel&&(a.channel=this.channel);try{var d=qd(a);c.postMessage(d,this.u)}catch(e){R(e,"WARNING")}}};
gi.prototype.j=function(){window.removeEventListener("message",this.l);L.prototype.j.call(this)};function hi(a,b,c){gi.call(this,a,b,c||Q("POST_MESSAGE_ORIGIN",void 0)||window.document.location.protocol+"//"+window.document.location.hostname,"widget");this.i=this.b=this.h=null}
n(hi,gi);function ii(){var a=this.c=new hi(!!Q("WIDGET_ID_ENFORCE")),b=z(this.La,this);a.h=b;a.i=null;this.c.channel="widget";if(a=Q("WIDGET_ID"))this.c.sessionId=a;this.g=[];this.i=!1;this.h={}}
h=ii.prototype;h.La=function(a,b,c){"addEventListener"==a&&b?(a=b[0],this.h[a]||"onReady"==a||(this.addEventListener(a,ji(this,a)),this.h[a]=!0)):this.la(a,b,c)};
h.la=function(){};
function ji(a,b){return z(function(a){this.sendMessage(b,a)},a)}
h.addEventListener=function(){};
h.sa=function(){this.i=!0;this.sendMessage("initialDelivery",this.Z());this.sendMessage("onReady");F(this.g,this.ka,this);this.g=[]};
h.Z=function(){return null};
function ki(a,b){a.sendMessage("infoDelivery",b)}
h.ka=function(a){this.i?this.c.sendMessage(a):this.g.push(a)};
h.sendMessage=function(a,b){this.ka({event:a,info:void 0==b?null:b})};
h.dispose=function(){this.c=null};function li(a){ii.call(this);this.b=a;this.f=[];this.addEventListener("onReady",z(this.Ja,this));this.addEventListener("onVideoProgress",z(this.Sa,this));this.addEventListener("onVolumeChange",z(this.Ta,this));this.addEventListener("onApiChange",z(this.Na,this));this.addEventListener("onPlaybackQualityChange",z(this.Pa,this));this.addEventListener("onPlaybackRateChange",z(this.Qa,this));this.addEventListener("onStateChange",z(this.Ra,this));this.addEventListener("onWebglSettingsChanged",z(this.Ua,
this))}
C(li,ii);h=li.prototype;h.la=function(a,b,c){if(this.b[a]){b=b||[];if(0<b.length&&Xh(a)){var d=b;if(y(d[0])&&!x(d[0]))d=d[0];else{var e={};switch(a){case "loadVideoById":case "cueVideoById":e=Zh.apply(window,d);break;case "loadVideoByUrl":case "cueVideoByUrl":e=Yh.apply(window,d);break;case "loadPlaylist":case "cuePlaylist":e=$h.apply(window,d)}d=e}b.length=1;b[0]=d}this.b.handleExternalCall(a,b,c);Xh(a)&&ki(this,this.Z())}};
h.Ja=function(){var a=z(this.sa,this);this.c.b=a};
h.addEventListener=function(a,b){this.f.push({eventType:a,listener:b});this.b.addEventListener(a,b)};
h.Z=function(){if(!this.b)return null;var a=this.b.getApiInterface();La(a,"getVideoData");for(var b={apiInterface:a},c=0,d=a.length;c<d;c++){var e=a[c],f=e;if(0==f.search("get")||0==f.search("is")){f=e;var g=0;0==f.search("get")?g=3:0==f.search("is")&&(g=2);f=f.charAt(g).toLowerCase()+f.substr(g+1);try{var k=this.b[e]();b[f]=k}catch(l){}}}b.videoData=this.b.getVideoData();b.currentTimeLastUpdated_=A()/1E3;return b};
h.Ra=function(a){a={playerState:a,currentTime:this.b.getCurrentTime(),duration:this.b.getDuration(),videoData:this.b.getVideoData(),videoStartBytes:0,videoBytesTotal:this.b.getVideoBytesTotal(),videoLoadedFraction:this.b.getVideoLoadedFraction(),playbackQuality:this.b.getPlaybackQuality(),availableQualityLevels:this.b.getAvailableQualityLevels(),currentTimeLastUpdated_:A()/1E3,playbackRate:this.b.getPlaybackRate(),mediaReferenceTime:this.b.getMediaReferenceTime()};this.b.getVideoUrl&&(a.videoUrl=
this.b.getVideoUrl());this.b.getVideoContentRect&&(a.videoContentRect=this.b.getVideoContentRect());this.b.getProgressState&&(a.progressState=this.b.getProgressState());this.b.getPlaylist&&(a.playlist=this.b.getPlaylist());this.b.getPlaylistIndex&&(a.playlistIndex=this.b.getPlaylistIndex());this.b.getStoryboardFormat&&(a.storyboardFormat=this.b.getStoryboardFormat());ki(this,a)};
h.Pa=function(a){ki(this,{playbackQuality:a})};
h.Qa=function(a){ki(this,{playbackRate:a})};
h.Na=function(){for(var a=this.b.getOptions(),b={namespaces:a},c=0,d=a.length;c<d;c++){var e=a[c],f=this.b.getOptions(e);b[e]={options:f};for(var g=0,k=f.length;g<k;g++){var l=f[g],m=this.b.getOption(e,l);b[e][l]=m}}this.sendMessage("apiInfoDelivery",b)};
h.Ta=function(){ki(this,{muted:this.b.isMuted(),volume:this.b.getVolume()})};
h.Sa=function(a){a={currentTime:a,videoBytesLoaded:this.b.getVideoBytesLoaded(),videoLoadedFraction:this.b.getVideoLoadedFraction(),currentTimeLastUpdated_:A()/1E3,playbackRate:this.b.getPlaybackRate(),mediaReferenceTime:this.b.getMediaReferenceTime()};this.b.getProgressState&&(a.progressState=this.b.getProgressState());ki(this,a)};
h.Ua=function(){var a={sphericalProperties:this.b.getSphericalProperties()};ki(this,a)};
h.dispose=function(){li.w.dispose.call(this);for(var a=0;a<this.f.length;a++){var b=this.f[a];this.b.removeEventListener(b.eventType,b.listener)}this.f=[]};function mi(a){a=void 0===a?!1:a;L.call(this);this.b=new N(a);bd(this,Ea(cd,this.b))}
C(mi,L);mi.prototype.subscribe=function(a,b,c){return this.c?0:this.b.subscribe(a,b,c)};
mi.prototype.h=function(a,b){this.c||this.b.J.apply(this.b,arguments)};function ni(a,b,c){mi.call(this);this.f=a;this.g=b;this.i=c}
C(ni,mi);function bi(a,b,c){if(!a.c){var d=a.f;d.c||a.g!=d.b||(a={id:a.i,command:b},c&&(a.data=c),d.b.postMessage(qd(a),d.g))}}
ni.prototype.j=function(){this.g=this.f=null;ni.w.j.call(this)};function oi(a,b,c){L.call(this);this.b=a;this.g=c;this.h=V(window,"message",z(this.i,this));this.f=new ni(this,a,b);bd(this,Ea(cd,this.f))}
C(oi,L);oi.prototype.i=function(a){var b;if(b=!this.c)if(b=a.origin==this.g)a:{b=this.b;do{b:{var c=a.source;do{if(c==b){c=!0;break b}if(c==c.parent)break;c=c.parent}while(null!=c);c=!1}if(c){b=!0;break a}b=b.opener}while(null!=b);b=!1}if(b&&(b=a.data,t(b))){try{b=JSON.parse(b)}catch(d){return}b.command&&(c=this.f,c.c||c.h("command",b.command,b.data,a.origin))}};
oi.prototype.j=function(){Xf(this.h);this.b=null;oi.w.j.call(this)};function pi(){var a=hb(qi),b;return Ed(new M(function(c,d){a.onSuccess=function(a){we(a)?c(a):d(new ri("Request failed, status="+a.status,"net.badstatus",a))};
a.onError=function(a){d(new ri("Unknown request error","net.unknown",a))};
a.L=function(a){d(new ri("Request timed out","net.timeout",a))};
b=Je("//googleads.g.doubleclick.net/pagead/id",a)}),function(a){a instanceof Fd&&b.abort();
return Bd(a)})}
function ri(a,b){E.call(this,a+", errorCode="+b);this.errorCode=b;this.name="PromiseAjaxError"}
n(ri,E);function si(){this.c=0;this.b=null}
si.prototype.then=function(a,b,c){return 1===this.c&&a?(a=a.call(c,this.b),wd(a)?a:ti(a)):2===this.c&&b?(a=b.call(c,this.b),wd(a)?a:ui(a)):this};
si.prototype.getValue=function(){return this.b};
si.prototype.$goog_Thenable=!0;function ui(a){var b=new si;a=void 0===a?null:a;b.c=2;b.b=void 0===a?null:a;return b}
function ti(a){var b=new si;a=void 0===a?null:a;b.c=1;b.b=void 0===a?null:a;return b}
;function vi(a){E.call(this,a.message||a.description||a.name);this.isMissing=a instanceof wi;this.isTimeout=a instanceof ri&&"net.timeout"==a.errorCode;this.isCanceled=a instanceof Fd}
n(vi,E);vi.prototype.name="BiscottiError";function wi(){E.call(this,"Biscotti ID is missing from server")}
n(wi,E);wi.prototype.name="BiscottiMissingError";var qi={format:"RAW",method:"GET",timeout:5E3,withCredentials:!0},xi=null;function me(){if("1"===bb(ke(),"args","privembed"))return Bd(Error("Biscotti ID is not available in private embed mode"));xi||(xi=Ed(pi().then(yi),function(a){return zi(2,a)}));
return xi}
function yi(a){a=a.responseText;if(0!=a.lastIndexOf(")]}'",0))throw new wi;a=JSON.parse(a.substr(4));if(1<(a.type||1))throw new wi;a=a.id;ne(a);xi=ti(a);Ai(18E5,2);return a}
function zi(a,b){var c=new vi(b);ne("");xi=ui(c);0<a&&Ai(12E4,a-1);throw c;}
function Ai(a,b){T(function(){Ed(pi().then(yi,function(a){return zi(b,a)}),va)},a)}
function Bi(){try{var a=w("yt.ads.biscotti.getId_");return a?a():me()}catch(b){return Bd(b)}}
;function Ci(a){if("1"!==bb(ke(),"args","privembed")){a&&le();try{Bi().then(function(a){a=oe(a);a.bsq=Di++;Pe("//www.youtube.com/ad_data_204",{wa:!1,B:a,withCredentials:!0})},function(){}),T(Ci,18E5)}catch(b){R(b)}}}
var Di=0;var Z=w("ytglobal.prefsUserPrefsPrefs_")||{};v("ytglobal.prefsUserPrefsPrefs_",Z,void 0);function Ei(){this.b=Q("ALT_PREF_COOKIE_NAME","PREF");var a=Bb.get(""+this.b,void 0);if(a){a=decodeURIComponent(a).split("&");for(var b=0;b<a.length;b++){var c=a[b].split("="),d=c[0];(c=c[1])&&(Z[d]=c.toString())}}}
h=Ei.prototype;h.get=function(a,b){Fi(a);Gi(a);var c=void 0!==Z[a]?Z[a].toString():null;return null!=c?c:b?b:""};
h.set=function(a,b){Fi(a);Gi(a);if(null==b)throw Error("ExpectedNotNull");Z[a]=b.toString()};
h.remove=function(a){Fi(a);Gi(a);delete Z[a]};
h.save=function(){xg(this.b,this.dump(),63072E3)};
h.clear=function(){for(var a in Z)delete Z[a]};
h.dump=function(){var a=[],b;for(b in Z)a.push(b+"="+encodeURIComponent(String(Z[b])));return a.join("&")};
function Gi(a){if(/^f([1-9][0-9]*)$/.test(a))throw Error("ExpectedRegexMatch: "+a);}
function Fi(a){if(!/^\w+$/.test(a))throw Error("ExpectedRegexMismatch: "+a);}
function Hi(a){a=void 0!==Z[a]?Z[a].toString():null;return null!=a&&/^[A-Fa-f0-9]+$/.test(a)?parseInt(a,16):null}
wa(Ei);var Ii=null,Ji=null,Ki=null,Li={};function Mi(a){var b=a.id;a=a.ve_type;var c=Jg++;a=new Hg({veType:a,veCounter:c,elementIndex:void 0,dataElement:void 0,youtubeData:void 0});Li[b]=a;b=Og();c=Ng();b&&c&&Rg(b,c,[a])}
function Ni(a){var b=a.csn;a=a.root_ve_type;if(b&&a&&(Pg(b,a),a=Ng()))for(var c in Li){var d=Li[c];d&&Rg(b,a,[d])}}
function Oi(a){Li[a.id]=new Hg({trackingParams:a.tracking_params})}
function Pi(a){var b=Og();a=Li[a.id];b&&a&&ug("visualElementGestured",{csn:b,ve:Ig(a),gestureType:"INTERACTION_LOGGING_GESTURE_TYPE_GENERIC_CLICK"},Dg,void 0,void 0)}
function Qi(a){a=a.ids;var b=Og();if(b)for(var c=0;c<a.length;c++){var d=Li[a[c]];d&&ug("visualElementShown",{csn:b,ve:Ig(d),eventType:1},Dg,void 0,void 0)}}
function Ri(){var a=Ii;a&&a.startInteractionLogging&&a.startInteractionLogging()}
;v("yt.setConfig",P,void 0);v("yt.config.set",P,void 0);v("yt.setMsg",Ze,void 0);v("yt.msgs.set",Ze,void 0);v("yt.logging.errors.log",Xe,void 0);
v("writeEmbed",function(){var a=Q("PLAYER_CONFIG",void 0);Ci(!0);"gvn"==a.args.ps&&(document.body.style.backgroundColor="transparent");var b=document.referrer,c=Q("POST_MESSAGE_ORIGIN");window!=window.top&&b&&b!=document.URL&&(a.args.loaderUrl=b);Q("LIGHTWEIGHT_AUTOPLAY")&&(a.args.autoplay="1");Ii=a=ph(a);a.addEventListener("onScreenChanged",Ni);a.addEventListener("onLogClientVeCreated",Mi);a.addEventListener("onLogServerVeCreated",Oi);a.addEventListener("onLogVeClicked",Pi);a.addEventListener("onLogVesShown",
Qi);a.addEventListener("onReady",Ri);b=Q("POST_MESSAGE_ID","player");Q("ENABLE_JS_API")?Ki=new li(a):Q("ENABLE_POST_API")&&t(b)&&t(c)&&(Ji=new oi(window.parent,b,c),Ki=new fi(a,Ji.f));c=je("BG_P");Nf(c)&&(Q("BG_I")||Q("BG_IU"))&&(Jf=!0,If.initialize(Q("BG_I",null),Q("BG_IU",null),c,Mf,void 0));Ef()},void 0);
v("yt.www.watch.ads.restrictioncookie.spr",function(a){Se(a+"mac_204?action_fcts=1");return!0},void 0);
var Si=se(function(){var a="ol";X.mark&&(0==a.lastIndexOf("mark_",0)||(a="mark_"+a),X.mark(a));a=Th();var b=O();a.ol&&(a._ol=a._ol||[a.ol],a._ol.push(b));a.ol=b;a=Ah();if(b=a.ol)Bf(b),a.ol=0;Ph().tick_ol=void 0;O();Qh()?(a=Rh(),uh("tick_ol_"+a)||rh("latencyActionTicked",{tickName:"ol",clientActionNonce:a},void 0),a=!0):a=!1;if(a=!a)a=!w("yt.timing.pingSent_");if(a&&(b=Q("TIMING_ACTION",void 0),a=Th(),w("ytglobal.timingready_")&&b&&a._start&&(b=Sh()))){Mh||(zh(Eh,new Ch(Math.round(b-a._start),void 0)),
Mh=!0);b=!0;var c=Q("TIMING_WAIT",[]);if(c.length)for(var d=0,e=c.length;d<e;++d)if(!(c[d]in a)){b=!1;break}b&&Vh()}a=Ei.getInstance();c=!!((Hi("f"+(Math.floor(119/31)+1))||0)&67108864);b=1<window.devicePixelRatio;document.body&&fd(document.body,"exp-invert-logo")&&(b&&!fd(document.body,"inverted-hdpi")?(d=document.body,d.classList?d.classList.add("inverted-hdpi"):fd(d,"inverted-hdpi")||(d.className+=0<d.className.length?" inverted-hdpi":"inverted-hdpi")):!b&&fd(document.body,"inverted-hdpi")&&gd());
c!=b&&(c="f"+(Math.floor(119/31)+1),d=Hi(c)||0,d=b?d|67108864:d&-67108865,0==d?delete Z[c]:(b=d.toString(16),Z[c]=b.toString()),a.save())}),Ti=se(function(){var a=Ii;
a&&a.sendAbandonmentPing&&a.sendAbandonmentPing();Q("PL_ATT")&&If.dispose();a=0;for(var b=Cf.length;a<b;a++)Bf(Cf[a]);Cf.length=0;vf("//static.doubleclick.net/instream/ad_status.js");Df=!1;P("DCLKSTAT",0);dd(Ki,Ji);if(a=Ii)a.removeEventListener("onScreenChanged",Ni),a.removeEventListener("onLogClientVeCreated",Mi),a.removeEventListener("onLogServerVeCreated",Oi),a.removeEventListener("onLogVeClicked",Pi),a.removeEventListener("onLogVesShown",Qi),a.removeEventListener("onReady",Ri),a.destroy();Li=
{}});
window.addEventListener?(window.addEventListener("load",Si),window.addEventListener("unload",Ti)):window.attachEvent&&(window.attachEvent("onload",Si),window.attachEvent("onunload",Ti));B("yt.abuse.player.botguardInitialized",w("yt.abuse.player.botguardInitialized")||Of);B("yt.abuse.player.invokeBotguard",w("yt.abuse.player.invokeBotguard")||Pf);B("yt.abuse.dclkstatus.checkDclkStatus",w("yt.abuse.dclkstatus.checkDclkStatus")||Ff);B("yt.player.exports.navigate",w("yt.player.exports.navigate")||Qg);
B("yt.util.activity.init",w("yt.util.activity.init")||$f);B("yt.util.activity.getTimeSinceActive",w("yt.util.activity.getTimeSinceActive")||cg);B("yt.util.activity.setTimestamp",w("yt.util.activity.setTimestamp")||ag);}).call(this);
