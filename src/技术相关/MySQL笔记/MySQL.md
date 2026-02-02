# MySQL笔记

### 一. MySQL服务的启动与停止以及客户端连接

1. 启动

   ```cmd
   net start mysql80   (这个是自己安装时候的服务名字，不区分大小写)
   ```

2. 停止

   ```cmd
   net stop mysql80   (这个是自己安装时候的服务名字，不区分大小写)
   ```

3. 客户端连接

   ```cmd
   mysql [-h 127.0.0.1] [-P 3306] -u root -p
   ```

### 二. DDL（Data Definition Language）语句

1. ###  DDL-数据库操作

   1. 查询

      查询所有数据库

      ```mysql
      SHOW DATABASES;
      ```

      查询当前数据库

      ```mysql
      SELECT DATABASE();
      ```

   2. 创建

      ```mysql
      CREATE DATABASE [IF NOT EXISTS] 数据库名 [DEFAULT CHARSET 字符集] [COLLATE 排序规则];
      ```

   3. 删除

      ```mysql
      DROP DATABASE [IF EXISTS] 数据库名;
      ```

   4. 使用

      ```mysql
      USE 数据库名;
      ```

2. ###  DDL-表操作-查询

   1. 查询当前数据库所有的表

      ```mysql
      SHOW TABLES;
      ```

   2. 查询表结构

      ```mysql
      DESC 表名;
      ```

   3. 查询指定表的建表语句

      ```mysql
      SHOW CREATE TABLE 表名;
      ```

3. ###  DDL-表操作-创建

   1. 创建表

      ```mysql
      CREATE TABLE 表名(
          字段1 字段1类型 [COMMENT 注释],
          字段2 字段2类型 [COMMENT 注释],
          字段3 字段3类型 [COMMENT 注释],
          字段4 字段4类型 [COMMENT 注释]
      ) [COMMENT 注释];
      ```

4. ### DDL-表操作-修改

   1. 添加字段

      ```mysql
      ALTER TABLE 表名 ADD 新字段名 新字段类型 [COMMENT 注释] [约束];
      ```

   2. 修改数据类型

      ```mysql
      ALTER TABLE 表名 MODIFY 字段名 新数据类型;
      ```

   3. 修改字段名和字段类型

      ```mysql
      ALTER TABLE 表名 CHANGE 旧字段名 新字段名 新字段类型(长度) [COMMENT 注释] [约束];
      ```

   4. 删除字段

      ```mysql
      ALTER TABLE 表名 DROP 字段名;
      ```

   5. 修改表名

      ```mysql
      ALTER TABLE 表名 RENAME TO 新表名;
      ```

   6. 删除表

      ```mysql
      -- 删除表
      DROP TABLE [IF EXISTS] 表名;
      -- 删除表后重新创建该表，相当于删除表中的数据
      TRUNCATE TABLE 表名;
      ```

### 三. DML（Data Manipulation Language）

1. ### DML-添加数据

   1. 给指定字段添加数据

      ```MYSQL
      INSERT INTO 表名(字段1, 字段2, ...) VALUES(值1, 值2, ...);
      ```

   2. 给全部字段添加数据

      ```mysql
      INSERT INTO 表名 VALUES(值1, 值2, ...);
      ```

   3. 批量添加数据

      ```mysql
      INSERT INTO 表名(字段1, 字段2, ...) VALUES(值1, 值2, ...), (值1, 值2, ...), (值1, 值2, ...);
      INSERT INTO 表名 VALUES(值1, 值2, ...), (值1, 值2, ...), (值1, 值2, ...);
      ```

2. ### DML-修改数据

   1. 基本语法

      ```mysql
      UPDATE 表名 SET 字段1=值1, 字段2=值2, ... [WHERE 条件];
      ```

3. ### DML-删除数据

   1. 基本语法

      ```MYSQL
      DELETE FROM 表名 [WHERE 条件];
      ```


### 四. DQL（Data Query Language）

1. ### DQL语法

   1. 基本结构

      ```mysql
      SELECT 
      	字段列表
      FROM 
      	表名列表
      WHERE
      	条件列表
      GROUP BY
      	分组字段列表
      HAVING
      	分组后条件列表
      ORDER BY
      	排序字段列表
      LIMIT
      	分页列表
      ```

2. ### DQL-基本查询

   1. 查询多个字段

      ```MYSQL
      SELECT 字段1 [AS 别名1], 字段2 [AS 别名1], ... FROM 表名;
      SELECT * FROM 表名;
      ```

   2. 去除重复记录

      ```MYSQL
      SELECT DISTINCT 字段1 [AS 别名1], 字段2 [AS 别名1], ... FROM 表名;
      ```

3. ### DQL-条件查询

   1. 语法

      ```mysql
      SELECT 字段列表 FROM 表名 WHERE 条件列表;
      ```

   2. 条件

      ![](.\imgs\1.png)

4. ### DQL-聚合函数

   1. 常见聚合函数

      ![](.\imgs\2.png)

   2. 语法

      ```mysql
      SELECT 聚合函数(字段列表) FROM 表名;
      ```

5. ### DQL-分组查询

   1. 语法

      ```mysql
      SELECT 字段列表 FROM 表名 [WHERE 条件] GROUP BY 分组字段名 [HAVING 分组后过滤条件];
      ```

6. ### DQL-排序查询

   1. 语法

      ```mysql
      SELECT 字段列表 FROM 表名 ORDER BY 字段1 排序方式1, 字段2 排序方式2;
      ```

   2. 排序方式

      ```mysql
      ASC:升序（默认值）
      DESC:降序
      ```

7. ### DQL-分页查询

   1. 语法（limit是方言）

      ```mysql
      SELECT 字段名 FROM 表名 LIMIT 起始索引, 查询记录数;
      ```

8. ### DQL-执行顺序

   1. 图示：

      ![](.\imgs\3.png)

### 五. DCL（Data Control Language）

1. ### DCL-管理用户

   1. 查询用户

      ```mysql
      -- 事实上是存放在 mysql 系统数据库中的 user 表中
      select * from mysql.user;
      ```

   2. 创建用户

      ```mysql
      create user '用户名'@'主机名' identified by '密码';
      -- 例子
      -- 只能够在当前主机访问
      create user 'new_user'@'localhost' identified by '123456';
      -- 能在任意主机访问
      create user 'new_user'@'%' identified by '123456';
      ```

   3. 修改用户密码

      ```mysql
      alter user '用户名'@'主机名' identified with mysql_native_password by '新密码';
      -- 例子
      alter user 'new_user'@'localhost' identified with mysql_native_password by '1234';
      ```

   4. 删除用户

      ```mysql
      drop user '用户名'@'主机名';
      ```

2. ### DCL-权限控制

   1. 常见的权限

      ![](.\imgs\4.png)

   2. 查询权限

      ```mysql
      show grants for 'new_user'@'localhost';
      ```

   3. 授予权限

      ```mysql
      grant all on 数据库名.表名 to 'new_user'@'localhost';
      ```

   4. 撤销权限

      ```mysql
      revoke all on 数据库名.表名 from 'new_user'@'localhost';
      ```

### 六. 函数

1. 字符串函数

   ![](.\imgs\5.png)

2. 数值函数

   ![](.\imgs\6.png)

3. 日期时间函数

   ![](.\imgs\7.png)

4. 流程控制函数

   ![](.\imgs\8.png)

### 七. 约束

1. 概述

   ![](.\imgs\9.png)

2. 添加外键约束

   ![](.\imgs\10.png)

3. 删除外键约束

   ```mysql
   alter table 表名 drop foreign key 外键名称;
   ```

4. 删除/更新行为

   ![](.\imgs\11.png)



