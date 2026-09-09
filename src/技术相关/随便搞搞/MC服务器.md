# 用docker搭建MC服务器（2025.6.4）

前言：默认服务器已经安装好docker。

### 一、拉取docker镜像

由于拉取官方docker镜像需要魔法，所以这里找到一个替代方案

````shell
# 网站：https://docker.aityp.com/image/docker.io/itzg/minecraft-server:latest
docker pull swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/itzg/minecraft-server:latest
# 查看已经拉取的镜像
docker images
````

### 二、创建docker容器

1.准备工作

```shell
# 创建文件夹并进入
mkdir <文件夹名称>
cd <文件夹名称>
# 创建yml配置文件
vim docker-compose.yml
```

2.创建好yml文件后，将以下内容复制进yml文件内

```dockerfile
services:
  mc:
    image: swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/itzg/minecraft-server
    tty: true
    stdin_open: true
    ports:
      - "25565:25565"
      - "25575:25575"
    environment:
      EULA: "TRUE"
      VERSION: "1.21.6"
    volumes:
      # attach the relative directory 'data' to the container's /data path
      - ./data:/data
```

3.创建并启动容器

```shell
# 启动！
docker compose up -d
# 关闭
docker compose down
```

操作完后，你的mc服务器已经启动了，就是那么简单，当然要记得去服务器上开放25565端口。

### 三、其他内容补充

1.server.properties 文件说明：

### 核心设置

1. **`difficulty=easy`**
   - 游戏难度：`easy`(简单)，可选 `peaceful`(和平), `normal`(普通), `hard`(困难)
2. **`gamemode=survival`**
   - 默认游戏模式：`survival`(生存)，可选 `creative`(创造), `adventure`(冒险), `spectator`(旁观)
3. **`level-name=world`**
   - 世界存档名称，对应数据目录中的 `world` 文件夹
4. **`max-players=20`**
   - 服务器最大玩家数：20人
5. **`motd=A Minecraft Server`**
   - 服务器描述（玩家在服务器列表中看到的描述）

------

### 网络与连接

1. **`server-port=25565`**
   - 服务器监听端口（与 Docker 端口映射匹配）
2. **`online-mode=false`**
   - **重要**：关闭正版验证（允许非正版玩家加入）
3. **`enable-rcon=true`**
   - 启用远程控制台(RCON)，通过 `rcon.port=25575` 访问
   - 密码：`rcon.password=df039071c9acbf9738bd03d6`（需保密！）
4. **`enable-status=true`**
   - 允许玩家查询服务器状态（显示在线玩家/最大玩家数）
5. **`network-compression-threshold=256`**
   - 网络数据压缩阈值（>256字节的数据包会被压缩）

------

### 游戏机制

1. **`pvp=true`**
   - 允许玩家间互相攻击
2. **`spawn-monsters=true`**
   - 生成敌对生物（怪物）
3. **`generate-structures=true`**
   - 生成自然建筑（村庄/地牢等）
4. **`allow-flight=false`**
   - 禁止玩家飞行（除非有鞘翅或药水效果）
5. **`view-distance=10`**
   - 玩家可视区块距离（影响性能）
6. **`spawn-protection=16`**
   - 出生点保护半径（16格内禁止非OP玩家破坏）

------

### 安全设置

1. **`enforce-secure-profile=true`**
   - 强制玩家使用经过验证的账号
2. **`white-list=false`**
   - 关闭白名单（所有玩家可自由加入）
3. **`enforce-whitelist=false`**
   - 不强制使用白名单
4. **`op-permission-level=4`**
   - OP玩家权限等级（4=最高权限）

------

### 性能优化

1. **`max-tick-time=60000`**
   - 单tick最大处理时间（60秒，避免卡顿）
2. **`max-chained-neighbor-updates=1000000`**
   - 连锁更新最大次数（防止连锁卡顿）
3. **`sync-chunk-writes=true`**
   - 同步区块写入（提高稳定性，降低性能）

------

### 其他重要设置

1. **`level-type=minecraft:normal`**
   - 世界类型：标准世界（可选 amplified(放大), flat(超平坦)）
2. **`hardcore=false`**
   - 关闭极限模式（死亡后无法重生）
3. **`allow-nether=true`**
   - 允许进入下界维度
4. **`player-idle-timeout=0`**
   - 无操作踢出时间（0=禁用）

2.rcon-cli的使用

本来有一种远程控制的方法，但是尝试了几次没有成功，所以这里介绍备用方案：直接进入容器控制

````shell
docker compose exec mc rcon-cli
# 其中 mc 是 yml 文件中容器的名字，通过该命令可以进入 rcon-cil 从而对服务器发送命令
````

下面介绍些常用命令：

1. **玩家管理**：
   - `list`：列出当前在线的玩家。
   - `kick <玩家名> [原因]`：将玩家踢出服务器。
   - `ban <玩家名> [原因]`：封禁玩家。
   - `pardon <玩家名>`：解除封禁。
   - `op <玩家名>`：授予玩家管理员权限。
   - `deop <玩家名>`：撤销玩家管理员权限。
   - `whitelist add <玩家名>`：将玩家加入白名单。
   - `whitelist remove <玩家名>`：从白名单中移除玩家。
   - `whitelist on/off`：启用/禁用白名单。
   - `whitelist list`：列出所有白名单玩家。
2. **世界管理**：
   - `save-all`：强制保存服务器世界（存档）。
   - `save-on`：开启自动保存（默认开启）。
   - `save-off`：关闭自动保存（通常在备份前使用，避免写入）。
   - `difficulty <peaceful/easy/normal/hard>`：设置游戏难度。
   - `gamemode <survival/creative/adventure/spectator> [玩家名]`：更改玩家游戏模式（若省略玩家名则更改自己）。
   - `time set <时间值>`：设置游戏时间（如`time set day`设置为白天，`time set 0`设置为清晨）。
   - `gamerule <规则名> <值>`：更改游戏规则（如`gamerule keepInventory true`设置死亡不掉落）。
3. **服务器控制**：
   - `stop`：安全关闭服务器（会保存世界）。
   - `reload`：重新加载服务器（重新读取配置和插件，可能会导致不稳定，慎用）。
   - `restart`：有些服务器支持重启命令（需插件支持）。
4. **信息查询**：
   - `tps`：显示服务器Tick速率（衡量服务器性能）。
   - `mspt`：显示服务器每tick的毫秒数（性能指标）。
   - `seed`：显示世界的种子（需有权限）。
5. **聊天与通信**：
   - `say <消息>`：以服务器名义广播消息（会以`[Server] 消息`的形式出现）。
   - `tell <玩家名> <消息>`：给指定玩家发送私信。



