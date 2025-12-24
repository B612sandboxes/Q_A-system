"""
Windows环境修复模块
解决LangChain在Windows上导入pwd模块的问题
"""

import sys
import platform

def fix_pwd_module():
    """修复pwd模块导入问题"""
    if platform.system() == "Windows":
        try:
            # 创建虚拟的pwd模块
            class PwdModule:
                @staticmethod
                def getpwuid(uid):
                    class Passwd:
                        pw_name = "user"
                        pw_passwd = ""
                        pw_uid = uid
                        pw_gid = 0
                        pw_gecos = "User"
                        pw_dir = "C:\\Users\\user"
                        pw_shell = "C:\\Windows\\System32\\cmd.exe"
                    return Passwd()
                
                @staticmethod
                def getpwnam(name):
                    return PwdModule.getpwuid(0)
            
            # 注入到sys.modules
            sys.modules['pwd'] = PwdModule()
            print("✅ pwd模块修复完成 (Windows环境)")
            return True
        except Exception as e:
            print(f"❌ pwd模块修复失败: {e}")
            return False
    else:
        # Linux/Mac不需要修复
        print("✅ 非Windows环境，无需修复pwd模块")
        return True


if __name__ == "__main__":
    fix_pwd_module()
    print("✅ 环境修复测试完成")