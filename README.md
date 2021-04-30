# webapi repository
template repository for frontend(vue) and backend(fastapi)

# build container

```
make
```

# setup frontend

## build vue environment

### restore vue ci environment

you should use this task after `git clone` this repository

```
make frontend-ci
```
### create project if you want

```
make frontend-init
```

### install libraries for project after init

```
make frontend-install
```

# use

## dev environment

```
make frontend-dev
```

## prod environment

```
make frontend-build
```

# setup backend

## startup webapi

```
make backend-webapi
```

## test for api

```
make backend-hello backend-post backend-test-request
```

```
{                                                                                                                       
  "type": "aaa",                                                                                                        
  "name": null,                                                                                                         
  "body": {                                                                                                             
    "mode": "pretrain",                                                                                                 
    "color": "red"                                                                                                      
  }                                                                                                                     
}                                                                                                                       
===== [http://localhost:8000/] =====                                                                                    
<!DOCTYPE html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1,shrink-to
-fit=no"><title>frontend</title><link href=/static/css/app.32cf90d21871ce14420020bdae23d491.css rel=stylesheet></head><b
ody><div id=app></div><script type=text/javascript src=/static/js/manifest.2ae2e69a05c33dfc65f8.js></script><script type
=text/javascript src=/static/js/vendor.673449851c94779368e7.js></script><script type=text/javascript src=/static/js/app.
062c2f2e6e54dcb19147.js></script></body></html>                                                                         
===== [http://localhost:8000/hello] =====                                                                               
{"Hello":"World!!!"}
===== [http://localhost:8000/front/static/a.html] =====
<html>
  <head>
    <title>たいとる✨</title>
  </head>
  <body>
    <form class="simple-form">
      <h2>ファイルアップロード</h2>
      <label for="title">タイトル</label>
      <input type="text" name="title">
      <button>投稿する</button>
    </form>
  </body>
</html>

===== [http://localhost:8000/static/a.html] =====
<html>
  <head>
    <title>たいとる✨</title>
  </head>
  <body>
    <form class="simple-form">
      <h2>ファイルアップロード</h2>
      <label for="title">タイトル</label>
      <input type="text" name="title">
      <button>投稿する</button>
    </form>
  </body>
</html>

```
