(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[7995],{52103:function(e,t,n){Promise.resolve().then(n.bind(n,81430))},81430:function(e,t,n){"use strict";n.r(t),n.d(t,{Error:function(){return a},default:function(){return c}});var r=n(27573),i=n(53603),o=n(7653);let l=e=>{let{href:t,children:n}=e;return(0,r.jsx)("a",{href:t,className:"pointer-events-auto mb-[60px] mt-0 w-[300px] text-center font-bold leading-[36px] text-white",style:{backgroundColor:"rgb(97, 87, 255)"},children:n})};var u=n(87657),s=n(54747);let a=()=>{let{isBot:e}=(0,o.useContext)(i.Z);return(0,r.jsx)(r.Fragment,{children:(0,r.jsx)(s.default,{children:(0,r.jsxs)("div",{className:"flex h-full w-full flex-col items-center justify-center py-[20px] pb-[50px]",children:[(0,r.jsx)("video",{className:"fixed top-0 h-[1015px] opacity-[0.23]",src:"/static/img/videos/tron.mp4",muted:!0,playsInline:!0,autoPlay:!0,loop:!0}),(0,r.jsx)("img",{src:"/static/img/error_pages/crying-cowbow-emoji.gif",alt:"Crying Cowboy Emoji",className:"isolate mt-[30px] h-[300px] w-[300px] opacity-100"}),(0,r.jsxs)("div",{className:"z-[1] my-[50px] mb-[20px] flex w-[400px] flex-col items-center text-center font-bold leading-[15px] opacity-100",children:[(0,r.jsx)("h4",{className:"isolate mb-[0px] text-[17px] opacity-100",style:{color:"white"},children:"Oops! There's nothing here."}),(0,r.jsx)("h4",{className:"mb-[7px] text-[17px]",style:{color:"white"},children:"For GIFs that DO exist, here's our trending feed..."})]}),(0,r.jsx)("div",{className:"ss-icon ss-navigatedown mb-[15px] text-white"}),e?null:(0,r.jsx)(u.Z,{}),(0,r.jsx)("div",{className:"pointer-events-none fixed bottom-0 flex h-[30vh] w-screen items-end justify-center bg-gradient-to-t from-black/80 via-black/0",children:(0,r.jsx)(l,{href:"/",children:"Continue to Our Homepage"})})]})})})};var c=a},87657:function(e,t,n){"use strict";var r=n(27573),i=n(64474),o=n(74101);let l=new(n(2159)).sF(o.publicRuntimeConfig.fourOhFourApiKey);t.Z=e=>{let{}=e;return(0,r.jsx)(i.ZP,{fetchGifs:e=>l.trending({offset:e})})}},64070:function(e,t,n){"use strict";n.d(t,{default:function(){return i.a}});var r=n(23842),i=n.n(r)},23842:function(e,t,n){"use strict";Object.defineProperty(t,"__esModule",{value:!0}),Object.defineProperty(t,"default",{enumerable:!0,get:function(){return o}});let r=n(11887);n(27573),n(7653);let i=r._(n(58379));function o(e,t){var n;let r={loading:e=>{let{error:t,isLoading:n,pastDelay:r}=e;return null}};"function"==typeof e&&(r.loader=e);let o={...r,...t};return(0,i.default)({...o,modules:null==(n=o.loadableGenerated)?void 0:n.modules})}("function"==typeof t.default||"object"==typeof t.default&&null!==t.default)&&void 0===t.default.__esModule&&(Object.defineProperty(t.default,"__esModule",{value:!0}),Object.assign(t.default,t),e.exports=t.default)},42972:function(e,t,n){"use strict";Object.defineProperty(t,"__esModule",{value:!0}),Object.defineProperty(t,"BailoutToCSR",{enumerable:!0,get:function(){return i}});let r=n(20951);function i(e){let{reason:t,children:n}=e;if("undefined"==typeof window)throw new r.BailoutToCSRError(t);return n}},58379:function(e,t,n){"use strict";Object.defineProperty(t,"__esModule",{value:!0}),Object.defineProperty(t,"default",{enumerable:!0,get:function(){return a}});let r=n(27573),i=n(7653),o=n(42972),l=n(69111);function u(e){return{default:e&&"default"in e?e.default:e}}let s={loader:()=>Promise.resolve(u(()=>null)),loading:null,ssr:!0},a=function(e){let t={...s,...e},n=(0,i.lazy)(()=>t.loader().then(u)),a=t.loading;function c(e){let u=a?(0,r.jsx)(a,{isLoading:!0,pastDelay:!0,error:null}):null,s=t.ssr?(0,r.jsxs)(r.Fragment,{children:["undefined"==typeof window?(0,r.jsx)(l.PreloadCss,{moduleIds:t.modules}):null,(0,r.jsx)(n,{...e})]}):(0,r.jsx)(o.BailoutToCSR,{reason:"next/dynamic",children:(0,r.jsx)(n,{...e})});return(0,r.jsx)(i.Suspense,{fallback:u,children:s})}return c.displayName="LoadableComponent",c}},69111:function(e,t,n){"use strict";Object.defineProperty(t,"__esModule",{value:!0}),Object.defineProperty(t,"PreloadCss",{enumerable:!0,get:function(){return o}});let r=n(27573),i=n(92399);function o(e){let{moduleIds:t}=e;if("undefined"!=typeof window)return null;let n=(0,i.getExpectedRequestStore)("next/dynamic css"),o=[];if(n.reactLoadableManifest&&t){let e=n.reactLoadableManifest;for(let n of t){if(!e[n])continue;let t=e[n].files.filter(e=>e.endsWith(".css"));o.push(...t)}}return 0===o.length?null:(0,r.jsx)(r.Fragment,{children:o.map(e=>(0,r.jsx)("link",{precedence:"dynamic",rel:"stylesheet",href:n.assetPrefix+"/_next/"+encodeURI(e),as:"style"},e))})}},93779:function(e,t,n){"use strict";Object.defineProperty(t,"__esModule",{value:!0});var r=n(7653);t.default=function(e){r.useEffect(e,[])}},98089:function(e,t,n){"use strict";Object.defineProperty(t,"__esModule",{value:!0});var r=n(83780),i=n(7653),o=r.__importDefault(n(45175));t.default=function(e){var t=i.useRef(0),n=i.useState(e),r=n[0],l=n[1],u=i.useCallback(function(e){cancelAnimationFrame(t.current),t.current=requestAnimationFrame(function(){l(e)})},[]);return o.default(function(){cancelAnimationFrame(t.current)}),[r,u]}},45175:function(e,t,n){"use strict";Object.defineProperty(t,"__esModule",{value:!0});var r=n(83780),i=n(7653),o=r.__importDefault(n(93779));t.default=function(e){var t=i.useRef(e);t.current=e,o.default(function(){return function(){return t.current()}})}},30138:function(e,t,n){"use strict";var r=n(83780),i=n(7653),o=r.__importDefault(n(98089)),l=n(11525);t.Z=function(e,t){void 0===e&&(e=1/0),void 0===t&&(t=1/0);var n=o.default({width:l.isBrowser?window.innerWidth:e,height:l.isBrowser?window.innerHeight:t}),r=n[0],u=n[1];return i.useEffect(function(){if(l.isBrowser){var e=function(){u({width:window.innerWidth,height:window.innerHeight})};return l.on(window,"resize",e),function(){l.off(window,"resize",e)}}},[]),r}}},function(e){e.O(0,[1307,9968,4551,664,7522,5569,3968,6566,9220,9729,9180,1293,1528,1744],function(){return e(e.s=52103)}),_N_E=e.O()}]);
//# sourceMappingURL=not-found-7d4aec1a85cfa511.js.map